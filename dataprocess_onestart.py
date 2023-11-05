import subprocess

py_execute = ["dataset/process_Tmall.py", "utils/graph.py"]
for py in py_execute:
    result = subprocess.run(["python", py],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    print(result.stdout)
    print(result.stderr)
    if result.returncode == 0:
        print(py + "executed successfully!")
    else:
        print(py + "process ended with executed code{}".format(result.returncode))
