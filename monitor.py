#!/usr/bin/env python3
"""Training monitor: checks training status every 120 seconds.
Detects NaN loss, OOM errors, frozen epochs, and restarts with fixes if needed.
"""

import os
import sys
import time
import subprocess
import re
import json
import signal


LOG_FILE = "runs/training.log"
TRAIN_SCRIPT = "train.py"
PID_FILE = "runs/train.pid"
CONFIG_FILE = "config/default.yaml"
MAX_RESTARTS = 5
CHECK_INTERVAL = 120  # seconds


def get_train_pid():
    """Get PID of running training process."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        try:
            os.kill(pid, 0)  # Check if process exists
            return pid
        except OSError:
            return None
    return None


def check_process_alive(pid):
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_training(pid):
    """Kill the training process."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        if check_process_alive(pid):
            os.kill(pid, signal.SIGKILL)
        print(f"[MONITOR] Killed training process {pid}")
    except OSError:
        pass


def parse_last_log_lines(n=20):
    """Parse the last n lines of the training log."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return []


def detect_issues(log_lines):
    """Detect training issues from log lines."""
    issues = []
    
    full_text = "".join(log_lines)
    
    # NaN loss
    if "nan" in full_text.lower() and "loss" in full_text.lower():
        issues.append("NaN_LOSS")
    
    # OOM
    if "CUDA out of memory" in full_text or "OutOfMemoryError" in full_text:
        issues.append("OOM")
    
    # RuntimeError
    if "RuntimeError" in full_text:
        issues.append("RUNTIME_ERROR")
    
    # Check for frozen training (same loss for multiple epochs)
    loss_vals = []
    for line in log_lines:
        match = re.search(r'train_loss:\s*([\d.]+)', line)
        if match:
            loss_vals.append(float(match.group(1)))
    
    if len(loss_vals) >= 5:
        last5 = loss_vals[-5:]
        if max(last5) - min(last5) < 1e-5:
            issues.append("FROZEN_LOSS")
    
    return issues


def fix_config_for_issue(issue):
    """Modify config to fix detected issues."""
    import yaml
    
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    
    if issue == "OOM":
        old_bs = cfg["data"]["batch_size"]
        new_bs = max(4, old_bs // 2)
        cfg["data"]["batch_size"] = new_bs
        print(f"[MONITOR] FIX: Reduced batch_size {old_bs} -> {new_bs}")
    
    elif issue == "NaN_LOSS":
        old_lr = cfg["training"]["learning_rate"]
        new_lr = old_lr * 0.5
        cfg["training"]["learning_rate"] = new_lr
        cfg["training"]["warmup_epochs"] = 10  # Longer warmup
        print(f"[MONITOR] FIX: Reduced LR {old_lr} -> {new_lr}, warmup=10")
    
    elif issue == "FROZEN_LOSS":
        old_lr = cfg["training"]["learning_rate"] 
        new_lr = old_lr * 2.0
        cfg["training"]["learning_rate"] = min(new_lr, 1e-2)
        print(f"[MONITOR] FIX: Increased LR {old_lr} -> {new_lr}")
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def start_training(resume=True):
    """Start training via nohup."""
    resume_arg = ""
    ckpt = "checkpoints/latest.pt"
    if resume and os.path.exists(ckpt):
        resume_arg = f"--resume {ckpt}"
    
    cmd = (f"cd {os.path.dirname(os.path.abspath(__file__))} && "
           f"source venv/bin/activate && "
           f"nohup python {TRAIN_SCRIPT} {resume_arg} "
           f"> runs/train_output.log 2>&1 & echo $!")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           executable='/bin/bash')
    pid = int(result.stdout.strip())
    
    with open(PID_FILE, 'w') as f:
        f.write(str(pid))
    
    print(f"[MONITOR] Started training with PID {pid}")
    return pid


def get_current_metrics():
    """Get latest metrics from training history."""
    hist_path = "runs/training_history.json"
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        if history.get("val_dice"):
            return {
                "epoch": len(history["val_dice"]),
                "val_dice": history["val_dice"][-1] if history["val_dice"] else 0,
                "val_miou": history["val_miou"][-1] if history.get("val_miou") else 0,
                "train_loss": history["train_loss"][-1] if history.get("train_loss") else 0,
            }
    return None


def main():
    print("=" * 60)
    print("TRAINING MONITOR (checking every 120s)")
    print("=" * 60)
    
    restart_count = 0
    
    # Initial start
    pid = get_train_pid()
    if pid is None:
        print("[MONITOR] No training process found, starting...")
        pid = start_training(resume=False)
    else:
        print(f"[MONITOR] Training already running (PID {pid})")
    
    while True:
        time.sleep(CHECK_INTERVAL)
        
        pid = get_train_pid()
        
        if pid and check_process_alive(pid):
            # Process is running - check for issues
            log_lines = parse_last_log_lines(30)
            issues = detect_issues(log_lines)
            
            if issues:
                print(f"[MONITOR] Issues detected: {issues}")
                
                for issue in issues:
                    if issue in ("NaN_LOSS", "OOM"):
                        print(f"[MONITOR] Critical issue: {issue}. Killing and restarting...")
                        kill_training(pid)
                        time.sleep(3)
                        fix_config_for_issue(issue)
                        restart_count += 1
                        
                        if restart_count > MAX_RESTARTS:
                            print(f"[MONITOR] Max restarts ({MAX_RESTARTS}) reached. Exiting.")
                            return
                        
                        pid = start_training(resume=(issue != "NaN_LOSS"))
                        break
                    elif issue == "FROZEN_LOSS":
                        print(f"[MONITOR] Warning: Loss appears frozen")
            else:
                metrics = get_current_metrics()
                if metrics:
                    print(f"[MONITOR] Epoch {metrics['epoch']} | "
                          f"val_dice={metrics['val_dice']:.4f} | "
                          f"val_miou={metrics['val_miou']:.4f} | "
                          f"train_loss={metrics['train_loss']:.4f}")
                else:
                    print(f"[MONITOR] Training running (PID {pid}), no metrics yet")
        else:
            # Process not running
            print("[MONITOR] Training process not running")
            
            # Check if it completed successfully
            log_lines = parse_last_log_lines(5)
            full_text = "".join(log_lines)
            
            if "Training complete" in full_text:
                print("[MONITOR] Training completed successfully!")
                metrics = get_current_metrics()
                if metrics:
                    print(f"[MONITOR] Final: val_dice={metrics['val_dice']:.4f}, "
                          f"val_miou={metrics['val_miou']:.4f}")
                return
            else:
                # Crashed - try to restart
                issues = detect_issues(parse_last_log_lines(50))
                if issues:
                    for issue in issues:
                        fix_config_for_issue(issue)
                
                restart_count += 1
                if restart_count > MAX_RESTARTS:
                    print(f"[MONITOR] Max restarts reached. Exiting.")
                    return
                
                print(f"[MONITOR] Restarting (attempt {restart_count}/{MAX_RESTARTS})...")
                pid = start_training(resume=True)


if __name__ == "__main__":
    main()
