# syntax=docker/dockerfile:1

# =============================================================================
# Estágio 1: builder — instala as dependências de runtime com uv
# =============================================================================
FROM python:3.11-slim AS builder

# Binário do uv (versão fixa, vindo da imagem oficial do Astral)
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# Copia só os manifestos de dependência primeiro (aproveita o cache do Docker)
COPY pyproject.toml uv.lock ./

# Instala APENAS as dependências de runtime, exatamente como resolvidas no
# uv.lock. --no-dev ignora o grupo dev (pytest, kagglehub, matplotlib...);
# --no-install-project não instala o pacote em si (a API roda via python main.py)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Estágio 2: runtime — imagem final enxuta, sem build tools e sem root
# =============================================================================
FROM python:3.11-slim AS runtime

# Usuário não-root (sem shell, reduz a superfície de ataque).
# O --chown nos COPY abaixo aplica a posse na extração, evitando um
# `chown -R` posterior (que duplicaria camadas grandes no overlayfs).
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copia o venv pronto do builder já com dono appuser
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copia o código da aplicação (o .dockerignore mantém lixo de fora)
COPY --chown=appuser:appuser . .

# Coloca o venv no PATH
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8075

CMD ["python", "main.py"]
