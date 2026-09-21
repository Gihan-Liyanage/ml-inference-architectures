class AppError(Exception):
    status_code = 500


class ModelLoadError(AppError):
    status_code = 503


class UnknownCategoryError(AppError):
    status_code = 422
