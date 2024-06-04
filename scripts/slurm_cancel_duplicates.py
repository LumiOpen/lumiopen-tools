import argparse
import subprocess
import re
from datetime import timedelta


def get_job_remain_time(job_id):
    command = f'squeue -o "%L" -j {job_id}'
    status, output = subprocess.getstatusoutput(command)
    if status != 0:
        raise subprocess.CalledProcessError(status, command, output)

    output_lines = output.split("\n")
    if len(output_lines) > 1 and output_lines[1]:
        return parse_time(output_lines[1])
    return None


def parse_time(time_str):
    match = re.match(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', time_str)
    if match:
        days, hours, minutes, seconds = match.groups()
        days = int(days) if days else 0
        return timedelta(days=days, hours=int(hours), minutes=int(minutes), seconds=int(seconds))
    return None


def get_running_jobs(job_name, owner):
    command = f'squeue -o "%i %T" -n {job_name} -u {owner}'
    status, output = subprocess.getstatusoutput(command)
    if status != 0:
        raise subprocess.CalledProcessError(status, command, output)

    running_jobs = []
    output_lines = output.split("\n")[1:]
    for line in output_lines:
        if line:
            job_id, state = line.split()
            if state == 'RUNNING':
                running_jobs.append(job_id)
    return running_jobs


def cancel_job(job_id):
    command = f'scancel {job_id}'
    status, ouput = subprocess.getstatusoutput(command)
    if status != 0:
        raise subprocess.CalledProcessError(status, command, output)
    return output


def main(job_name, dry_run, owner):
    running_jobs = get_running_jobs(job_name, owner)
    print(f"Running jobs for {job_name}: {running_jobs}")
    if not running_jobs:
        return
    elif len(running_jobs) == 1:
        print(f"Only one job matching, nothing to do.")
        return
    
    running_jobs.sort(key=lambda x: int(x))
    job_to_keep = running_jobs[0]  # oldest job = smallest number
    max_remain_time = get_job_remain_time(job_to_keep)

    if max_remain_time < timedelta(hours=12):
        for job_id in running_jobs[1:]:
            job_remain_time = get_job_remain_time(job_id)
            if job_remain_time > max_remain_time:
                max_remain_time = job_remain_time
                job_to_keep = job_id
    print(f"Keeping job {job_id} with {max_remain_time} remaining")

    for job_id in running_jobs:
        if job_id == job_to_keep:
            continue
        if dry_run:
            print(f"Would cancel job {job_id}")
        else:
            print(f"Cancelling job {job_id}")
            cancel_job(job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage running jobs")
    parser.add_argument("job_name", help="Name of the job to manage")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    parser.add_argument("--owner", required=True, help="Filter jobs by owner")

    args = parser.parse_args()

    main(job_name=args.job_name, dry_run=args.dry_run, owner=args.owner)
