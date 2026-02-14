# Rollout checklist: ANALYSIS_PIPELINE_VERSION=v2

1. В Railway откройте сервис и добавьте переменную:
   - `ANALYSIS_PIPELINE_VERSION=v2`
2. Перезапустите деплой (или дождитесь автодеплоя из GitHub).
3. Прогоните 5-10 реальных звонков и проверьте:
   - в AmoCRM/Telegram нет цитат/evidence и сырых JSON;
   - в заметке есть строка `Участники: N` (когда есть диаризация);
   - поля `cost/payment_terms/date` не галлюцинируют при отсутствии данных.
4. В логах проверьте:
   - `v2 speaker stats: ...`
   - `v2 verify field=... status=... confidence=... evidence=...`
5. Если качество устраивает, оставьте `v2` включенным.
   Если нужен быстрый откат, верните `ANALYSIS_PIPELINE_VERSION=v1`.
