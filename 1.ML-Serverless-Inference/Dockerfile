# Build stage
FROM public.ecr.aws/lambda/python:3.12 AS builder

# Install poetry
ENV POETRY_VERSION=2.1.1 \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false

RUN pip install "poetry==${POETRY_VERSION}"

# Copy poetry files
WORKDIR ${LAMBDA_TASK_ROOT}
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Final stage
FROM public.ecr.aws/lambda/python:3.12

# Copy only the installed packages from builder
COPY --from=builder /var/lang/lib/python3.12/site-packages/ /var/lang/lib/python3.12/site-packages/

# Copy application code
WORKDIR ${LAMBDA_TASK_ROOT}
COPY . .

# Set the CMD to your handler
CMD [ "handler.run" ]
