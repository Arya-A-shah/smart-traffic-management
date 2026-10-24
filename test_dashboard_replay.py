import ast
from pathlib import Path


def _render_function_calls():
    source = Path('dashboard/app.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'render':
            return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'fit']
    raise AssertionError('render() function not found in dashboard/app.py')


def test_live_replay_plotly_calls_use_unique_keys():
    calls = _render_function_calls()
    plot_calls = [
        call for call in calls
        if any(isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == 'st' and arg.attr == 'plotly_chart'
               for arg in call.args[:1])
    ]
    assert plot_calls, 'No replay plotly_chart calls found in render()'
    assert all(any(keyword.arg == 'key' for keyword in call.keywords) for call in plot_calls)
