# Como executar
1. **(Windows) Criar ambiente (necessário python 3.10 em razão do Tensorflow)**
```bash
    py -3.10 -m venv .venv
```
2. **Ativar ambiente**
```bash
    .\.venv\Scripts\activate
```
3. **Instalar pacotes necessários**
```bash
    pip install -r requirements.txt
```

4.**Treine ou execute o modelo pronto**
**Para treinar**
```bash
    py .\main.py
```
**Para testar (Cheque os comentários no código)**
```bash
    py .\model_run.py
```