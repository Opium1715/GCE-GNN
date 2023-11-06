import subprocess

train_times = 5
py_execute = ["main.py"]
for py in py_execute:
    for i in range(train_times):
        result = subprocess.run(["python", py],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode == 0:
            print(py + " executed successfully in {} times! ".format(i+1))
        else:
            print(py + " process ended with executed code{} ".format(result.returncode))
