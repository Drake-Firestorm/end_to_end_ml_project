import sys

from end_to_end_ml_project_yt.logger import logging


def error_message_details(error, error_details: sys):
    _, _, exc_tb = error_details.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_messge = f"Error occured in python script {file_name} line number {exc_tb.tb_lineno} error message {error!s}"

    return error_messge


class CustomException(Exception):
    def __init__(self, error_messge, error_details: sys):
        super().__init__(error_messge)
        self.error_messge = error_message_details(error_messge, error_details)

    def __str__(self):
        return self.error_messge


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)
