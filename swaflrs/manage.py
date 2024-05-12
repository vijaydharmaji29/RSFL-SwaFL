#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import time
import psutil
import threading
import csv

cpu_ram_usage = []

def measure_cpu_ram():
    global cpu_ram_usage

    while True:
        time_now = time.time()
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        cpu_ram_usage.append((time_now, cpu_usage, ram_usage))
        time.sleep(1)

def write_analysis():
    while True:
        csv_file = "aggregator_cpu_ram_stats.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Timestamp', 'CPU Usage', 'RAM Usage'])
            # Write the data
            for row in cpu_ram_usage:
                writer.writerow(row)
    
        time.sleep(10)
    


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'swaflrs.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    thread = threading.Thread(target=measure_cpu_ram)
    thread.start()

    thread2 = threading.Thread(target=write_analysis)
    thread2.start()
    main()
