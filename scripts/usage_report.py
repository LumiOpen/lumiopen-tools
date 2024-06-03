import subprocess
import re
from datetime import datetime, timedelta

# chatgpt special

def parse_time_to_hours(time_str):
    """Converts sacct time format to total hours."""
    parts = time_str.split('-')
    if len(parts) == 2:
        days = int(parts[0])
        time = parts[1]
    else:
        days = 0
        time = parts[0]
    hours, minutes, seconds = map(int, time.split(':'))
    total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
    return total_hours

def parse_gpu_hours(data):
    """Parses the sacct output to calculate total GPU hours."""
    total_gpu_hours = 0
    for line in data.split('\n'):
        parts = line.split('|')
        if len(parts) < 4:
            continue
        user, elapsed, alloc_tres = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if not user:  # Skip lines without a username
            continue
        gpu_count = re.search(r'gres/gpu[^=]*=(\d+)', alloc_tres)
        if gpu_count:
            gpu_count = int(gpu_count.group(1))
            hours = parse_time_to_hours(elapsed)
            # gpu hours are for 2 GCDs, so should be divided by 2 from the gpu
            # count
            total_gpu_hours += (hours * gpu_count) / 2
    return total_gpu_hours

def execute_sacct_command(project_id, starttime, endtime):
    """Executes the sacct command and returns its output."""
    cmd = [
        'sacct', '-a', '-A', project_id,
        '--starttime', starttime,
        '--endtime', endtime,
        '--format', 'User,Elapsed,AllocTRES,Start', '-P'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout
    else:
        raise Exception("sacct command failed: " + result.stderr)

def get_month_start_end(year_month):
    """Returns the start and end date for the given year-month."""
    start_date = datetime.strptime(year_month + "-01", "%Y-%m-%d")
    next_month = start_date + timedelta(days=32)
    end_date = datetime(next_month.year, next_month.month, 1)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def main(projects, months):
    project_ids = projects.split(',')
    year_months = months.split(',')
    for year_month in year_months:
        total_gpu_hours = 0
        start_date, end_date = get_month_start_end(year_month)
        for project_id in project_ids:
            try:
                output = execute_sacct_command(project_id.strip(), start_date, end_date)
                month_gpu_hours = parse_gpu_hours(output)
                total_gpu_hours += month_gpu_hours
                print(f"GPU hours for project {project_id} during {year_month}: {month_gpu_hours}")
            except Exception as e:
                print(f"Error processing project {project_id} during {year_month}: {str(e)}")
        print(f"Total GPU hours for all projects during {year_month}: {total_gpu_hours}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python script.py <project_ids> <YYYY-MM,YYYY-MM>")

