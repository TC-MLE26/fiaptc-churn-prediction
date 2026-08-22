# Usa uma imagem oficial do Python, versão slim (mais leve)
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de dependência primeiro (para aproveitar o cache do Docker)
COPY requirements.txt .

# Instala as dependências (sem usar cache para deixar a imagem menor)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o container
COPY . .

# Expõe a porta que a aplicação vai rodar
EXPOSE 8075

# Comando para iniciar a API
CMD ["python", "main.py"]
