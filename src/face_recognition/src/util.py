import os
import psutil
import GPUtil
from threading import Thread
import time

class ResourceMonitor(Thread):
    def __init__(self, interval=3):
        Thread.__init__(self)
        self.interval = interval
        self.stopped = False
        self.process = psutil.Process(os.getpid())

        
    def run(self):
        while not self.stopped:
            cpu_usage = self.process.cpu_percent(interval=1)            
            memory_info = self.process.memory_info()
            memory_usage = memory_info.rss / psutil.virtual_memory().total * 100            
            gpu_usage = self.get_gpu_usage()
            
            print(f"Process CPU Usage: {cpu_usage:.2f}%, "
                  f"Process Memory Usage: {memory_usage:.2f}%, "
                  f"Process GPU Usage: {gpu_usage:.2f}%")
            
            time.sleep(self.interval)
            
    def get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,nounits,noheader'])
                for line in result.decode().strip().split('\n'):
                    pid, used_memory = map(int, line.split(','))
                    if pid == self.process.pid:
                        return (used_memory / gpus[0].memoryTotal) * 100
            return 0
        except Exception as e:
            print(f"Error getting GPU usage: {e}")
            return 0

    def stop(self):
        self.stopped = True
        
