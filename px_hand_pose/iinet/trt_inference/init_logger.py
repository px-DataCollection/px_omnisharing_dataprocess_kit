import os
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime
import glob

def get_logger(name='my_logger', save_dir=None, log_lvl='ERROR', max_bytes=1024 * 1024 * 10, backup_count=3, total_backup_count=10,
               fmt="%(asctime)s - %(name)s - %(levelname)s -  %(filename)s:%(lineno)d - %(message)s",
               print_to_console=False,
               *args, **kwargs):
    try:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(log_lvl)

        # Remove the oldest log files if the number of log files under save_dir exceeds backup_count
        if save_dir:
            file_list = glob.glob(os.path.join(save_dir, f"{name}_*"))
            file_list = [f for f in file_list if os.path.isfile(f)]
            if len(file_list) > total_backup_count:
                file_list.sort(key=os.path.getmtime)
                for i in range(len(file_list) - total_backup_count):
                    os.remove(file_list[i])

            # Create file handler with a dynamic filename based on timestamp
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            log_file = os.path.join(save_dir, f"{name}_{timestamp_str}")
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(file_handler)

        if print_to_console:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(console_handler)

        logger.info(f"Logger {name} created with level {log_lvl}")
        return logger
    except Exception:
        logger = logging.getLogger(name)
        logger.error(f"Error in get_logger: {traceback.format_exc()}")
        return logger
