# 📋 User Stories — Football Offside Detector App

Formato: Como [utilizador], quero [ação], para [benefício].
Critérios de Aceitação (CA) seguem cada US.

---

## 🗂️ ÉPICO 1 — Upload e Processamento de Vídeo

### US-01 — Upload de clip de vídeo
**Como** utilizador da app,
**quero** fazer upload de um clip de vídeo de um jogo de futebol,
**para** analisar lances de possível fora de jogo.

**Critérios de Aceitação:**
- [ ] Aceita formatos MP4, MOV e AVI até 500MB
- [ ] Mostra barra de progresso durante o upload
- [ ] Valida que o ficheiro é um vídeo antes de processar
- [ ] Mostra mensagem de erro clara se o formato não for suportado

---

### US-02 — Processamento automático do vídeo
**Como** utilizador da app,
**quero** que o vídeo seja processado automaticamente após o upload,
**para** não ter de configurar parâmetros técnicos manualmente.

**Critérios de Aceitação:**
- [ ] O processamento começa automaticamente após upload
- [ ] Mostra estimativa de tempo restante
- [ ] Deteta jogadores, árbitros e bola sem configuração manual
- [ ] Processa vídeos até 90 segundos em menos de 3 minutos

---

### US-03 — Calibração do campo (perspetiva)
**Como** utilizador da app,
**quero** marcar 4 pontos do campo no vídeo de forma simples,
**para** que a transformação de perspetiva seja precisa.

**Critérios de Aceitação:**
- [ ] Interface drag-and-drop para colocar 4 pontos no frame
- [ ] Preview em tempo real da transformação bird's eye view
- [ ] Opção de calibração automática (deteção das linhas do campo)
- [ ] Guarda a calibração para reutilizar em vídeos do mesmo estádio

---

## 🗂️ ÉPICO 2 — Visualização do Resultado

### US-04 — Ver o vídeo anotado com linha de fora de jogo
**Como** utilizador da app,
**quero** ver o vídeo original com a linha de fora de jogo desenhada,
**para** perceber visualmente qual a posição dos jogadores.

**Critérios de Aceitação:**
- [ ] Linha amarela vertical no frame no momento do passe
- [ ] Jogadores em fora de jogo destacados a vermelho
- [ ] Label "FORA DE JOGO" visível no frame
- [ ] Player ID visível por cima de cada jogador

---

### US-05 — Bird's eye view 2D em tempo real
**Como** utilizador da app,
**quero** ver um mini-campo 2D visto de cima em simultâneo com o vídeo,
**para** ter uma perspetiva clara das posições relativas de todos os jogadores.

**Critérios de Aceitação:**
- [ ] Mini-campo no canto inferior direito do vídeo
- [ ] Cada equipa com cor diferente (azul vs vermelho)
- [ ] Linha de fora de jogo desenhada no campo 2D
- [ ] Jogadores em fora de jogo destacados no campo 2D

---

### US-06 — Navegar frame a frame no lance
**Como** utilizador da app,
**quero** avançar e recuar frame a frame no momento do lance,
**para** analisar com precisão o momento exato do passe.

**Critérios de Aceitação:**
- [ ] Slider de navegação por frame
- [ ] Botões de avanço/recuo frame a frame
- [ ] Número do frame e timestamp visíveis
- [ ] Pausa automática no frame do lance detetado

---

## 🗂️ ÉPICO 3 — Gestão de Lances

### US-07 — Exportar vídeo anotado
**Como** utilizador da app,
**quero** exportar o vídeo processado com as anotações,
**para** partilhar o resultado com outros treinadores ou árbitros.

**Critérios de Aceitação:**
- [ ] Exporta em MP4 (H.264)
- [ ] Opção de exportar apenas o clip do lance (±5 segundos)
- [ ] Qualidade mínima de 720p no output
- [ ] Download disponível imediatamente após processamento

---

### US-08 — Histórico de lances analisados
**Como** utilizador registado,
**quero** ver o histórico de todos os lances que analisei,
**para** revisitar análises anteriores sem ter de reprocessar.

**Critérios de Aceitação:**
- [ ] Lista de lances com thumbnail, data e jogo
- [ ] Filtro por data, equipa ou resultado (fora de jogo / não)
- [ ] Cada lance mostra o veredito (offside/onside)
- [ ] Possibilidade de apagar lances do histórico

---

### US-09 — Partilhar análise com link
**Como** utilizador da app,
**quero** gerar um link de partilha para uma análise específica,
**para** enviar a análise a colegas sem precisarem de conta.

**Critérios de Aceitação:**
- [ ] Link público com validade de 7 dias
- [ ] O destinatário pode ver o vídeo e o campo 2D sem conta
- [ ] Opção de link privado (protegido por password)

---

## 🗂️ ÉPICO 4 — Conta e Planos

### US-10 — Registo e login
**Como** novo utilizador,
**quero** criar uma conta com email e password,
**para** guardar o meu histórico de análises.

**Critérios de Aceitação:**
- [ ] Registo com email + password ou Google OAuth
- [ ] Verificação de email obrigatória
- [ ] Recuperação de password por email
- [ ] Login persistente (token com 30 dias)

---

### US-11 — Plano gratuito vs premium
**Como** utilizador da app,
**quero** perceber as diferenças entre o plano gratuito e premium,
**para** decidir se vale a pena subscrever.

**Critérios de Aceitação:**
- [ ] Plano gratuito: 5 análises/mês, sem histórico
- [ ] Plano premium: análises ilimitadas, histórico, exportação HD, partilha
- [ ] Página de pricing clara com comparação lado a lado
- [ ] Upgrade disponível a qualquer momento via Stripe

---

### US-12 — Notificação quando a análise estiver pronta
**Como** utilizador,
**quero** receber uma notificação quando o processamento terminar,
**para** não precisar de ficar à espera na app.

**Critérios de Aceitação:**
- [ ] Notificação push (web) quando o processamento termina
- [ ] Email de notificação opcional (configurável)
- [ ] Notificação com link direto para o resultado

---

## 📊 Resumo por Épico

| Épico | US | Prioridade |
|---|---|---|
| Upload e Processamento | US-01, 02, 03 | 🔴 Must Have |
| Visualização | US-04, 05, 06 | 🔴 Must Have |
| Gestão de Lances | US-07, 08, 09 | 🟡 Should Have |
| Conta e Planos | US-10, 11, 12 | 🟡 Should Have |
