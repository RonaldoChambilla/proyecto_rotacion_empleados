# 📈 Predicción de Rotación de Empleados (Employee Churn) - Web App

Este repositorio contiene una aplicación web simple para predecir la probabilidad de que un empleado abandone la empresa (rotación), utilizando un modelo de Red Neuronal entrenado con TensorFlow/Keras.

## 🚀 Cómo Funciona (Funcionamiento Esperado)

1.  **Entrada de Datos:** El usuario ingresa información clave del empleado (edad, ingreso mensual, cargo, etc.) a través de un formulario HTML (`index.html`).
2.  **Preprocesamiento en el Navegador:** Los datos ingresados son preprocesados en JavaScript (`script.js`) utilizando parámetros (medias, desviaciones estándar, categorías) exportados desde el entrenamiento en Python (`preprocessor_params.json`). Esto asegura que los datos estén en el formato correcto para el modelo.
3.  **Predicción Local:** El modelo de Red Neuronal (cargado desde `modelo_attrition_web/`) realiza la predicción directamente en el navegador del usuario.
4.  **Resultado:** La aplicación muestra si el empleado tiene un "Riesgo alto de rotación" o "Riesgo bajo de rotación".

## 🛠️ Ejecución Local

Para ver la aplicación web:

1.  Clona o descarga este repositorio.
2.  Abre el archivo `index.html` en tu navegador web (Chrome, Firefox, Edge, etc.).

**Nota:** Es posible que para la carga de modelos locales, necesites un servidor web simple (ej. `python -m http.server` en la raíz del proyecto) si experimentas problemas de seguridad CORS, aunque usualmente funciona abriendo el `index.html` directamente.

## ⚠️ Desafío / Error Conocido

Al intentar cargar el modelo en el navegador, es posible que encuentres el siguiente error en la consola del desarrollador (F12):

An InputLayer should be passed


**Explicación:** Este error indica una incompatibilidad entre la forma en que el modelo de Keras fue guardado (usando Keras 3.x, que es la versión predeterminada con TensorFlow 2.19.0+ en entornos como Google Colab) y la versión de TensorFlow.js que se carga en el navegador (`4.22.0`). A pesar de usar la API Funcional de Keras y el formato `SavedModel` para la exportación, parece haber un desafío de compatibilidad persistente con la gestión de las capas de entrada del modelo entre estas versiones. Es un problema común en la evolución de librerías de ML.

---