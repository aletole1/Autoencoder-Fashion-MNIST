# Configuración 0

```json
{
    "learning_rate": 0.001,
    "dropout": 0.2,
    "batch_size": 100,
    "epochs": 60,
    "S2F": false
}
```

![](./F2S_0.001.png)

Pérdida de entrenamiento incorrecto: 0.008336843142751604

Pérdida de entrenamiento: 0.00542466553238531

Pérdida de validación: 0.0058524398179724815

# Configuración 1

```json
{
    "learning_rate": 0.01,
    "dropout": 0.2,
    "batch_size": 100,
    "epochs": 60,
    "S2F": false
}
```

![](./F2S_0.01.png)

Pérdida de entrenamiento incorrecto: 0.013036622072880467

Pérdida de entrenamiento: 0.009111385280266403

Pérdida de validación: 0.00955254609696567

# Configuración 2

```json
{
    "learning_rate": 0.001,
    "dropout": 0.2,
    "batch_size": 100,
    "epochs": 60,
    "S2F": true
}
```

![](./S2F_0.001.png)

Pérdida de entrenamiento incorrecto: 0.009528970294632018

Pérdida de entrenamiento: 0.006002399455755949

Pérdida de validación: 0.006420961203984916

# Configuración 3

```json
{
    "learning_rate": 0.01,
    "dropout": 0.2,
    "batch_size": 100,
    "epochs": 60,
    "S2F": true
}
```

![](./S2F_0.01.png)

Pérdida de entrenamiento incorrecto: 0.01559569428053995

Pérdida de entrenamiento: 0.011431274247976641

Pérdida de validación: 0.011714352313429117

