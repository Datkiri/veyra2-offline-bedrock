import ast
import io
import tokenize
from pathlib import Path

root=Path(__file__).resolve().parent.parent
count=0
for path in (root/'tools').glob('*.py'):
    text=path.read_text(encoding='utf-8')
    lines=text.splitlines(keepends=True)
    ranges=[]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type==tokenize.COMMENT:
            ranges.append((token.start[0]-1,token.start[1],token.end[0]-1,token.end[1]))
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and node.body:
            first=node.body[0]
            if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str):
                ranges.append((first.lineno-1,0,first.end_lineno-1,len(lines[first.end_lineno-1].rstrip('\r\n'))))
    for start,start_col,end,end_col in reversed(sorted(ranges)):
        for row in range(start,end+1):
            a=start_col if row==start else 0
            b=end_col if row==end else len(lines[row].rstrip('\r\n'))
            lines[row]=lines[row][:a]+' '*(b-a)+lines[row][b:]
    result=''.join(lines)
    ast.parse(result)
    path.write_text(result,encoding='utf-8')
    count+=len(ranges)
print({'removedPythonCommentsAndDocstrings':count})
