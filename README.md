Hlavní kód použit pro trénink modelů pro účely závěrečné práce: <br />

NLP laboratoř postavená na frameworku PyTorch Lightning, určená primárně pro binární klasifikaci. Architektura je plně modulární a umožňuje snadné přidávání dalších modelů, samplerů a vlastních ztrátových funkcí. Konfiguraci řídí nástroj Hydra, díky kterému lze parametry definovat buď v YAML souborech, nebo zadávat přímo přes příkazovou řádku ve formátu `key=value`.<br />

Při spouštění jakéhokoliv skriptu musí být vždy zvolen odpovídající model a "data lane". Například při přípravě dat pro konvoluční síť (CNN) je nutné nastavit parametr `data: CDL`. Stejně tak při načítání checkpointu `BiLSTM.ckpt` je nutné mít nastaveno `model: bilstm`.

===============================================<br />

A PyTorch Lightning-based NLP laboratory designed primarily for binary classification tasks. The architecture is fully modular, allowing for the easy integration of new models, samplers, and custom loss functions. Configuration logic is managed by Hydra, enabling you to define settings via YAML files or directly through the command-line interface using `key=value` arguments.<br />

When executing any script, the corresponding model and data processing lane must strictly match. For example, when preparing data for a CNN, you must set `data: CDL`. Similarly, when loading a `BiLSTM.ckpt` checkpoint, you must ensure `model: bilstm` is explicitly set in your configuration.<br />



