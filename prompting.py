import requests
import json
from typing import List, Dict, Any

try:
    from LibPath import insertPath
    insertPath()
except:
    pass


import inspect
class Array:
    def map(self, func):
        if type(func) == list:
            self._funcs = func
            return Array(list(map(self.funcToIter, self.array)))
        return Array(list(map(func, self.array)))
    def __init__(self, arr):
        self.array = arr
    def funcToIter(self, ele):
        res = ele
        for fun in self._funcs:
            res = fun(res)
        return res
    def filter(self, func):
        if type(func) == list:
            self._funcs = func
            return Array(list(filter(self.funcToIter, self.array)))
        return Array(list(filter(func, self.array)))
    def index(self):
        return Array(list(zip(range(len(self.array)), self.array)))
    def reshape(self, size):
        col = len(self.array) // size
        return Array(ListDB.reshape(self.array, (col, size)))
    def count(self):
        return len(self.array)
    def sum(self):
        return sum(self.array)
    def join(self, sep):
        return sep.join(self.array)
    def toDict(self, keyFunc, valueFunc):
        return Dictt({keyFunc(k): valueFunc(k) for k in self.array})
class NameSpace:
    pass
class ObjectOps:
    def make_obj():
        return NameSpace()
    def setEvenIfItdoesNotExist(obj, loc, val):
        if len(loc) == 0:
            return
        newLoc = loc[:-1]
        innerObj = obj
        for key in newLoc:
            if not ObjectOps.exists(innerObj, [key]):
                ObjectOps.setter(innerObj, [key], ObjectOps.make_obj())
            innerObj = getattr(innerObj, key)
        key = loc[-1]
        setattr(innerObj, key, val)
    def exists(obj, loc):
        val = obj
        for key in loc:
            if not hasattr(val, key):
                return False
            val = getattr(val, key)
        return True
    def setter(obj, loc, val):
        if len(loc) == 0:
            return
        innerObj = ObjectOps.getter(obj, loc[:-1])
        key = loc[-1]
        setattr(innerObj, key, val)
    def getter(obj, loc):
        val = obj
        for key in loc:
            val = getattr(val, key)
        return val
class ObjMaker:
    def namespace():
        return ObjectOps.make_obj()
    def variablesAndFunction(dictVals, ignoring=[], obj=None, ignoreIfExistsInObj=False):
        if obj is None:
            obj = ObjectOps.make_obj()
        for key in dictVals:
            if key in ignoring:
                continue
            if hasattr(obj, key) and ignoreIfExistsInObj:
                continue
            val = dictVals[key]
            if inspect.isclass(val):
                pass
            elif inspect.isfunction(val):
                ObjectOps.setEvenIfItdoesNotExist(obj, ['handlers', key], val)
                ObjectOps.setEvenIfItdoesNotExist(obj, ['handlers', 'defs', key], val)
            else:
                ObjectOps.setEvenIfItdoesNotExist(obj, ['process', key], val)
        ObjectOps.setEvenIfItdoesNotExist(obj, ['local_states'], dictVals)
        return obj
def LanguageParserV2():
    """Multi syntax parser. Can handle all types"""
    content = ''
    langComps = {'[': ']', '{': '}', '(': ')', "'": "'", '"': '"'}
    langCompsReversed = {langComps[k]: k for k in langComps}
    i = 0
    windowSize = 1
    def set_lang_components(comps):
        s.process.langComps = comps
        s.process.langCompsReversed = {comps[k]: k for k in comps}
        s.process.sizes = set(map(len, comps)).union(set(map(len, s.process.langCompsReversed)))
    def get_node(typ=''):
        node = ObjMaker.namespace()
        node.start = s.process.i
        node.end = node.start + s.process.windowSize
        node.value = s.process.content[node.start:node.end]
        node.typ = typ
        node.children = []
        return node
    def pop(queue):
        if len(queue) > 1:
            queue.pop()
    def parse_with_parent_child():
        root = s.handlers.get_node()
        queue = [root]
        s.process.i = 0
        s.process.windowSize = 1
        while s.process.i < len(s.process.content):
            a, b = s.handlers.existsInLang()
            if a or b:
                current = queue[-1]
                if a and b:
                    if current.value == s.process.c:
                        current.end = s.process.i + s.process.windowSize
                        s.handlers.pop(queue)
                    else:
                        node = s.handlers.get_node(s.process.c)
                        current.children.append(node)
                        queue.append(node)
                elif a:
                    node = s.handlers.get_node(s.process.c)
                    current.children.append(node)
                    queue.append(node)
                elif b:
                    if s.process.langCompsReversed[s.process.c] == current.typ:
                        current.end = s.process.i + s.process.windowSize
                        s.handlers.pop(queue)
            s.process.i += s.process.windowSize
        return root
    def existsInLang():
        for ws in s.process.sizes:
            s.process.windowSize = ws
            s.process.c = s.process.content[s.process.i:s.process.i + ws]
            if s.process.c in s.process.langComps:
                return (True, s.process.c in s.process.langCompsReversed)
            if s.process.c in s.process.langCompsReversed:
                return (False, True)
        s.process.windowSize = 1
        return (False, False)
    def _exists():
        s.process.c = s.process.content[s.process.i]
        return s.process.c in s.process.langCompsReversed
    def bfs(root, condition, allRes=False):
        queue = [root]
        res = []
        while True:
            node = queue.pop()
            if condition(node):
                if allRes:
                    res.append(node)
                else:
                    return node
            for ch in node.children:
                queue.append(ch)
            if len(queue) == 0:
                break
        return res
    s = ObjMaker.variablesAndFunction(locals())
    return s
def prepare_prompt(trains, test) -> str:
    prompt = "Given the following sample data:\n\n"
    for i, train_input in enumerate(trains):
        prompt += f"Train Input {i+1}:\nInput:\n{train_input['input']}\nOutput:\n{train_input['output']}\n\n"
    prompt += f"Test Case:\n{test}\n\n"
    prompt += f"""Please generate Python code to process the test case based on the pattern in the training data. The code should take the test case as input and return the expected output.

Your response should be in the following format:

```python
def process_test_case(test_case):
    # Your code here
    return result

# Example usage:
# result = process_test_case({test})
# print(result)
```

Ensure that your code is complete, correct, and follows this exact format."""
    return prompt
def generate_code(prompt: str, model: str = "codellama:7b") -> str:
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        raise Exception(f"Error generating code: {response.status_code} - {response.text}")
def extract_function(code: str) -> str:
    # Extract the function definition using regex
    langParser = LanguageParserV2()
    langParser.handlers.set_lang_components({'```python': '```'})
    content = File.getFileContent('test.txt')
    langParser.process.content = content
    root = langParser.handlers.parse_with_parent_child()
    return Array(root.children).map(lambda x: langParser.process.content[x.start + 9:x.end - 3].strip()).filter(
        lambda x: "def process_test_case" in x).array[0]
def execute_generated_code(code: str, test_case: List[List[int]]) -> Any:
    function_code = extract_function(code)
    local_namespace = {}
    exec(function_code, globals(), local_namespace)
    process_test_case = local_namespace['process_test_case']
    return process_test_case(test_case)
def sampleFromData(data, key):
    return data[key]["train"], data[key]["test"][0]["input"]
def loadData():
    from FileDatabase import File
    import json
    data = json.loads(File.getFileContent(r"C:\Users\rajab\Downloads\arc-agi_evaluation_challenges.json\arc-agi_evaluation_challenges.json"))
    return data
def checkOnRandomSamples(data, sampleSize = 10):
    from tqdm import tqdm
    import random
    samples = random.sample(sorted(data), sampleSize,)
    results = {}
    for sample in tqdm(samples):
        results[sample] = {}
        trains, test = sampleFromData(data, sample)
        prompt = prepare_prompt(trains, test)
        generated_code = generate_code(prompt)
        results[sample]["inputs"] = {"prompts": prompt, "code": generated_code}
        try:
            result = execute_generated_code(generated_code, test)
            results[sample]["result"] = {"status": "success", "msg": "", "result": result}
            results[sample]["result"]["correct-on-train-inputs"] = trains[0]["output"] == execute_generated_code(generated_code, trains[0]["input"])
        except Exception as e:
            results[sample]["result"] = {"status": "failure", "msg": str(e), "error": e}
    return results
    
    
# res = checkOnRandomSamples(loadData(), sampleSize=2)