"""C02 全量复核：新内核执行 Notebook、运行脚本并与原生 PyTorch 对照。

python C02/verify.py --report /tmp/c02-verification.json
默认不改学习文件；--notebook-output 可另存执行后的 Notebook。
仅在已有环境中运行 CPU 前向计算，无下载、安装、训练或模型 API 调用。
"""

import argparse
import contextlib
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import platform
import runpy
import sys
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent


def execute_notebook(output_path):
    source = ROOT / 'C02/lesson01.ipynb'
    notebook = nbformat.read(source, as_version=4)
    persisted_code_cells = sum(cell.cell_type == 'code' for cell in notebook.cells)
    marker = '__C02_KERNEL_PROBE__'
    probe = nbformat.v4.new_code_cell(
        'import sys, json, platform\n'
        f'print({marker!r} + json.dumps({{'
        '"python_executable": sys.executable, "python": platform.python_version(), '
        '"torch": torch.__version__, "checks": verification_checks, '
        '"expected_errors": expected_errors}, ensure_ascii=False))'
    )
    notebook.cells.append(probe)
    with TemporaryDirectory(prefix='c02-kernel-') as directory:
        manager = KernelManager(kernel_name='python3', connection_file=str(Path(directory) / 'kernel.json'))
        # 明确使用运行本验证器的解释器，不依赖 kernelspec 的显示名。
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(notebook, km=manager, timeout=120, allow_errors=False)
        try:
            client.execute(cwd=str(ROOT / 'C02'))
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    texts = ''.join(item.get('text', '') for item in notebook.cells[-1].outputs)
    evidence = json.loads(next(line[len(marker):] for line in texts.splitlines() if line.startswith(marker)))
    assert Path(evidence['python_executable']).resolve() == Path(sys.executable).resolve()
    assert all(evidence['checks'].values())
    notebook.cells.pop()
    errors = [output for cell in notebook.cells if cell.cell_type == 'code' for output in cell.get('outputs', []) if output.output_type == 'error']
    assert not errors
    if output_path:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
    return {
        'method': 'fresh Jupyter kernel; all cells in order; allow_errors=False; existing interpreter explicitly selected',
        'persisted_code_cells_executed': persisted_code_cells, 'temporary_probe_cells': 1,
        'unhandled_cell_errors': len(errors), **evidence,
    }


def execute_script(relative):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(ROOT / relative), run_name='__main__')


def valid_mask(case):
    return case['valid'] if 'valid' in case else case['valid_mask']


def clear_pad(tensor, valid):
    return tensor.masked_fill(~valid.unsqueeze(-1), 0.0)


def compare_single(case):
    # functional SDPA 的布尔 Mask：True=允许，与本课 allowed 一致。
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(
            case['query'].unsqueeze(1), case['key'].unsqueeze(1), case['value'].unsqueeze(1),
            attn_mask=case['allowed'].unsqueeze(1), dropout_p=0.0,
        ).squeeze(1)
        expected = clear_pad(expected, valid_mask(case))
    torch.testing.assert_close(expected, case['output'], atol=1e-6, rtol=1e-5)
    return {'output_matches': True, 'shape': list(expected.shape)}


def compare_multi(case):
    reference = nn.MultiheadAttention(case['D'], case['H'], dropout=0, bias=False, batch_first=True)
    length = case['h'].shape[1]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([case[name].weight for name in ('q_proj', 'k_proj', 'v_proj')]))
        reference.out_proj.weight.copy_(case['out_proj'].weight)
        expected, weights = reference(
            case['h'], case['h'], case['h'], attn_mask=~causal,
            key_padding_mask=~valid_mask(case), average_attn_weights=False,
        )
        expected = clear_pad(expected, valid_mask(case))
    torch.testing.assert_close(expected, case['output'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(weights, case['weights'], atol=1e-6, rtol=1e-5)
    # 实际 Q 投影的拆头合并往返也必须保持数值顺序。
    query = case['query']
    merged_query = query.transpose(1, 2).contiguous().view_as(case['h'])
    torch.testing.assert_close(merged_query, case['q_proj'](case['h']))
    return {'output_matches': True, 'per_head_weights_match': True, 'split_merge_roundtrip': True, 'shape': list(expected.shape)}


def compare_block(case):
    reference = nn.TransformerEncoderLayer(
        d_model=case['D'], nhead=case['H'], dim_feedforward=case['D_ff'],
        dropout=0, activation='gelu', batch_first=True, norm_first=True,
    )
    layers = vars(case['block'])['_modules'] if 'block' in case else case
    length = case['h'].shape[1]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    with torch.no_grad():
        reference.self_attn.in_proj_weight.copy_(torch.cat([layers[name].weight for name in ('q_proj', 'k_proj', 'v_proj')]))
        reference.self_attn.in_proj_bias.zero_()
        reference.self_attn.out_proj.weight.copy_(layers['out_proj'].weight)
        reference.self_attn.out_proj.bias.zero_()
        for name, layer in [('norm1', reference.norm1), ('norm2', reference.norm2), ('ff_up', reference.linear1), ('ff_down', reference.linear2)]:
            layer.load_state_dict(layers[name].state_dict())
        expected = reference(case['h'], src_mask=~causal, src_key_padding_mask=~valid_mask(case))
        expected = clear_pad(expected, valid_mask(case))
    torch.testing.assert_close(expected, case['output'], atol=1e-6, rtol=1e-5)
    return {'output_matches': True, 'shape': list(expected.shape)}


def extra_behavior_checks(case):
    run = case['run_block']
    h, valid = case['h'], case['valid']
    with torch.no_grad():
        base, _ = run(h, valid)
        changed = h.clone()
        changed[1, -1] += torch.linspace(-1.5, 1.5, case['D'])
        result, _ = run(changed, valid)
        torch.testing.assert_close(result[1, :-1], base[1, :-1], atol=1e-6, rtol=1e-5)
        u = case['trace']['after_attn'].detach()
        changed_u = u.clone()
        changed_u[1, 1] += torch.linspace(-2.0, 2.0, case['D'])
        ffn = lambda x: case['ff_down'](case['activation'](case['ff_up'](case['norm2'](x))))
        a, b = ffn(u), ffn(changed_u)
        unchanged = torch.ones_like(valid)
        unchanged[1, 1] = False
        torch.testing.assert_close(a[unchanged], b[unchanged])
        assert not torch.allclose(a[1, 1], b[1, 1])
        parameters = [case['out_proj'].weight, case['ff_down'].weight, case['ff_down'].bias]
        backups = [parameter.clone() for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.zero_()
            identity, _ = run(h, valid)
            torch.testing.assert_close(identity, h, atol=0, rtol=0)
        finally:
            for parameter, backup in zip(parameters, backups):
                parameter.copy_(backup)
        try:
            run(h, torch.zeros_like(valid))
        except ValueError as error:
            assert '没有可读 Key' in str(error)
        else:
            raise AssertionError('全屏蔽行应在 Softmax 前被拒绝')
    return {'causal_prefix_isolated': True, 'ffn_positionwise': True, 'zero_branches_preserve_residual': True, 'all_masked_rows_rejected': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--notebook-output', type=Path)
    args = parser.parse_args()
    notebook = execute_notebook(args.notebook_output)
    sources = [
        'C02/lesson02.py', 'C02/lesson03.py', 'C02/exercise/lesson03exercise.py',
        'C02/lesson04.py', 'C02/exercise/lesson04exercise.py',
        'C02/lesson05.py', 'C02/exercise/lesson05exercise.py',
    ]
    states = {relative: execute_script(relative) for relative in sources}
    l2, l3, e3 = (states[name] for name in sources[:3])
    l4, e4, l5, e5 = (states[name] for name in sources[3:])
    comparisons = {
        'lesson02_teacher': compare_single(l2), 'lesson02_homework': compare_single(l2['homework']),
        'lesson03_teacher_H2': compare_multi(l3), 'lesson03_homework_H3': compare_multi(e3['case_h3']),
        'lesson03_homework_H4': compare_multi(e3['case_h4']),
        'lesson04_teacher_Dff16': compare_block(l4), 'lesson04_homework_Dff24': compare_block(e4['result_24']),
        'lesson04_homework_Dff48': compare_block(e4['result_48']),
        'lesson05_teacher': compare_block(l5), 'lesson05_homework': compare_block(e5['result']),
    }
    for checks in [l2['checks'], e3['case_h3']['checks'], e3['case_h4']['checks'], e3['debug_checks'], l4['checks'], e4['result_24']['checks'], e4['result_48']['checks'], e5['result']['checks']]:
        assert all(checks.values())
    assert e4['layernorm_result']['matches'] and set(e4['expected_errors']) == {'A', 'B'}
    extras = extra_behavior_checks(l4)
    report = {
        'module': 'C02 Transformer foundations',
        'date': datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat(),
        'timezone': 'Asia/Shanghai', 'result': 'passed',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'nbformat': nbformat.__version__, 'nbclient': __import__('nbclient').__version__, 'device': 'cpu'},
        'notebook': notebook,
        'python_scripts': {relative: 'passed; expected errors are caught by the lesson' for relative in sources},
        'source_sha256': {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in ['C02/lesson01.ipynb', *sources, 'C02/verify.py']},
        'lesson02_checks': l2['checks'], 'lesson02_expected_errors': l2['error_results'],
        'lesson03_homework_checks': {str(h): e3[f'case_h{h}']['checks'] for h in (3, 4)},
        'lesson03_debug_checks': e3['debug_checks'],
        'lesson04_checks': l4['checks'],
        'lesson04_homework_checks': {str(width): e4[f'result_{width}']['checks'] for width in (24, 48)},
        'layernorm_manual': e4['layernorm_result'], 'lesson04_expected_errors': e4['expected_errors'],
        'lesson05_teacher_deltas': {name: l5[name] for name in ('causal_delta', 'open_delta', 'pad_delta', 'unmasked_delta')},
        'lesson05_homework_shapes': e5['result']['shapes'], 'lesson05_homework_checks': e5['result']['checks'],
        'lesson05_homework_deltas': e5['result']['deltas'], 'lesson05_debug_checks': e5['debug_result'],
        'native_reference_comparisons': comparisons, 'native_reference_case_count': len(comparisons),
        'reference_conditions': 'Same parameters, CPU, dropout=0; GELU and norm_first=True for Blocks; manual attention projections have no bias, reference attention biases set to zero; PAD outputs cleared on both sides. Boolean masks are converted for each API.',
        'comparison_tolerance': {'atol': 1e-6, 'rtol': 1e-5}, 'additional_behavior_checks': extras,
        'scope': 'Current fixed teaching examples, saved learning files and labeled assistant reference supplements. User requested answers as private completion confirmation. Passing reference code is not independent proof of learner mastery or language understanding.',
        'training_performed': False, 'dependencies_changed': False, 'paid_model_api_calls': 0,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'result': report['result'], 'notebook_code_cells': notebook['persisted_code_cells_executed'], 'notebook_checks': len(notebook['checks']), 'python_scripts': len(sources), 'native_reference_cases': len(comparisons), 'teacher_deltas': report['lesson05_teacher_deltas'], 'homework_deltas': report['lesson05_homework_deltas']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
