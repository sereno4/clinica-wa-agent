cat > README.md << 'EOF'
# 🏥 ClinicAI Pro

Sistema de atendimento automatizado de nível empresarial para clínicas e consultórios via WhatsApp. Construído com arquitetura Multi-Agente, Inteligência Artificial modular e foco extremo em estabilidade e anti-ban.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)

---

## ✨ Funcionalidades Principais

- 🧠 **Multi-Modelo de IA:** Escolha entre GPT-4o, Gemini 1.5 ou Grok por unidade. Controle fino de Temperatura.
- 🛡️ **Anti-Alucinação (RAG Strict):** O sistema só responde com base no que está indexado. Se não sabe, ele diz que não sabe. Controle de distância vetorial (< 1.2).
- 📚 **Base de Conhecimento Acumulativa:** Aceita PDF, TXT, CSV e JSON (FAQ). Fatiamento inteligente por parágrafos ou pares de Pergunta/Resposta.
- 📸 **Galeria de Mídia Automática:** Cadastre fotos com tags (ex: `rosto, resultado`). Se o cliente pedir, a IA envia a imagem automaticamente e avisa que está enviando.
- 🎙️ **Transcrição de Áudio Nativa:** Clientes enviam áudio no WhatsApp e a IA transcreve e responde via Whisper (OpenAI).
- 🛑 **Fila Anti-Ban do Meta:** Controle rigoroso de envio (1 msg a cada 4s por sessão). Ignora mensagens com mais de 2 minutos (evita spam de quedas).
- ⏰ **Consciência Temporal Absoluta:** Sabe a data exata, dia da semana, se é final de semana e corrige o cliente suavemente se ele disser "Bom dia" à noite.
- 🕵️ **Extração de Entidades (NLP):** Extrai datas e horários das conversas silenciosamente para geração de relatórios futuros.

---

## 🏗️ Arquitetura

O projeto é dividido em dois contêineres orquestrados pelo Docker Compose:

1. **Backend (FastAPI - Porta 8080):** Gerencia os agentes, banco vetorial (ChromaDB), memória, arquivos e a comunicação com a OpenAI/Gemini.
2. **WAHA (Porta 3000):** Motor não-oficial de conexão com o WhatsApp. Lida com QR Code, envio de mensagens e recepção de Webhooks.

---

## 🚀 Deploy em Produção (VPS / Ocean Digital)

O sistema foi 100% containerizado para facilitar seu deploy.

### 1. Pré-requisitos no Servidor
- Docker e Docker Compose instalados.
- Portas `8080` e `3000` liberadas no Firewall.

### 2. Clonar o repositório
```bash
git clone git@github.com:sereno4/clinica-wa-agent.git
cd clinica-wa-agent


📁 Estrutura de Armazenamento
Para manter tudo portátil, o sistema usa pastas locais mapeadas no Docker:

database/agentes/: JSONs com as configs de cada clínica.
database/historico/: Logs de conversas de cada cliente.
database/chroma_db/: Banco vetorial permanente.
kb_files/: Armazena temporariamente os PDFs/CSVs enviados pelo painel.
🛠️ Tecnologias Utilizadas
Backend: Python, FastAPI, Uvicorn
IA: OpenAI (GPT-4o/Whisper), Google Gemini, xAI Grok
Banco Vetorial: ChromaDB com all-MiniLM-L6-v2
WhatsApp Engine: WAHA (Devlikeapro)
Frontend: TailwindCSS, JavaScript Vanilla
Infraestrutura: Docker, Docker Compose


Desenvolvido para resolver problemas reais de atendimento no Brasil.

