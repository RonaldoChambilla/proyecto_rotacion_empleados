// Variable global para almacenar el modelo cargado
let model;

// Función asincrónica para cargar el modelo de TensorFlow.js
async function loadModel() {
    console.log("Cargando modelo...");
    try {
        // La ruta al modelo.json debe ser relativa a donde está index.html
        model = await tf.loadLayersModel('./modelo_attrition_web/model.json');
        console.log("Modelo cargado exitosamente.");
    } catch (error) {
        console.error("Error al cargar el modelo:", error);
        document.getElementById('result').innerText = "Error: No se pudo cargar el modelo. Verifique la consola para más detalles.";
    }
}

// Llama a la función para cargar el modelo cuando la página se cargue
window.onload = loadModel;

// Seleccionar el formulario y el div de resultado
const predictionForm = document.getElementById('predictionForm');
const resultDiv = document.getElementById('result');

predictionForm.addEventListener('submit', async (event) => {
    event.preventDefault(); // Evitar que el formulario se envíe de forma predeterminada

    if (!model) {
        resultDiv.innerText = "El modelo aún no está cargado. Por favor, espere.";
        return;
    }

    // 1. Recopilar datos del formulario
    const age = parseInt(document.getElementById('age').value);
    const monthlyIncome = parseInt(document.getElementById('monthlyIncome').value);
    const jobRole = document.getElementById('jobRole').value;
    const totalWorkingYears = parseInt(document.getElementById('totalWorkingYears').value);
    const yearsAtCompany = parseInt(document.getElementById('yearsAtCompany').value);
    const overTime = document.getElementById('overTime').value;
    const businessTravel = document.getElementById('businessTravel').value;

    // console.log("Datos del formulario:", { age, monthlyIncome, jobRole, totalWorkingYears, yearsAtCompany, overTime, businessTravel });

    // 2. Preprocesar los datos (replicar one-hot encoding de Python)
    // Inicializar un objeto con todas las columnas esperadas por el modelo,
    // con valores por defecto de 0 (para las dummies)
    // NOTA IMPORTANTE: Estas columnas y su orden DEBEN coincidir exactamente
    // con las columnas de tu X_train después del one-hot encoding en Python,
    // EXCLUYENDO 'Attrition_Yes'.
    // Puedes verificar el orden y los nombres de las columnas en Python
    // usando `X_train.columns` después del preprocesamiento.
    // He incluido las que suelen generarse, pero deberías verificarlo.

    const inputFeatures = {
        Age: age,
        DailyRate: 0, // No está en el formulario, asumimos un valor por defecto o 0
        DistanceFromHome: 0, // No está en el formulario, asumimos un valor por defecto o 0
        Education: 0, // No está en el formulario, asumimos un valor por defecto o 0
        EnvironmentSatisfaction: 0, // No está en el formulario, asumimos un valor por defecto o 0
        HourlyRate: 0, // No está en el formulario, asumimos un valor por defecto o 0
        JobInvolvement: 0, // No está en el formulario, asumimos un valor por defecto o 0
        JobLevel: 0, // No está en el formulario, asumimos un valor por defecto o 0
        JobSatisfaction: 0, // No está en el formulario, asumimos un valor por defecto o 0
        MonthlyIncome: monthlyIncome,
        MonthlyRate: 0, // No está en el formulario, asumimos un valor por defecto o 0
        NumCompaniesWorked: 0, // No está en el formulario, asumimos un valor por defecto o 0
        PercentSalaryHike: 0, // No está en el formulario, asumimos un valor por defecto o 0
        PerformanceRating: 0, // No está en el formulario, asumimos un valor por defecto o 0
        RelationshipSatisfaction: 0, // No está en el formulario, asumimos un valor por defecto o 0
        StandardHours: 80, // Generalmente constante en el dataset IBM
        StockOptionLevel: 0, // No está en el formulario, asumimos un valor por defecto o 0
        TotalWorkingYears: totalWorkingYears,
        TrainingTimesLastYear: 0, // No está en el formulario, asumimos un valor por defecto o 0
        WorkLifeBalance: 0, // No está en el formulario, asumimos un valor por defecto o 0
        YearsAtCompany: yearsAtCompany,
        YearsInCurrentRole: 0, // No está en el formulario, asumimos un valor por defecto o 0
        YearsSinceLastPromotion: 0, // No está en el formulario, asumimos un valor por defecto o 0
        YearsWithCurrManager: 0, // No está en el formulario, asumimos un valor por defecto o 0
        
        // One-hot encoded variables:
        // BusinessTravel
        'BusinessTravel_Travel_Frequently': 0,
        'BusinessTravel_Travel_Rarely': 0,
        // Department
        'Department_Research & Development': 0,
        'Department_Sales': 0,
        // JobRole
        'JobRole_Human Resources': 0,
        'JobRole_Laboratory Technician': 0,
        'JobRole_Manager': 0,
        'JobRole_Manufacturing Director': 0,
        'JobRole_Research Director': 0,
        'JobRole_Research Scientist': 0,
        'JobRole_Sales Executive': 0,
        'JobRole_Sales Representative': 0,
        // OverTime
        'OverTime_Yes': 0
    };

    // Actualizar valores de las variables one-hot encoded según la entrada del formulario
    if (businessTravel === 'Travel_Frequently') {
        inputFeatures['BusinessTravel_Travel_Frequently'] = 1;
    } else if (businessTravel === 'Travel_Rarely') {
        inputFeatures['BusinessTravel_Travel_Rarely'] = 1;
    }
    // No necesitamos manejar 'Non-Travel' explícitamente si drop_first=True en Python,
    // ya que su ausencia (ambas columnas a 0) ya lo representa.

    // JobRole: Establecer la columna correspondiente a 1
    if (jobRole) {
        const jobRoleColumn = `JobRole_${jobRole}`;
        if (inputFeatures.hasOwnProperty(jobRoleColumn)) { // Verificar si la columna existe
            inputFeatures[jobRoleColumn] = 1;
        } else {
            console.warn(`La columna para JobRole "${jobRole}" no fue encontrada. Asegúrate de que coincida con el preprocesamiento de Python.`);
            // Manejar caso donde el JobRole no es reconocido por el modelo
            // Podrías lanzar un error o simplemente dejarlo como 0.
        }
    }

    if (overTime === 'Yes') {
        inputFeatures['OverTime_Yes'] = 1;
    }
    // Si overTime es 'No', 'OverTime_Yes' permanece en 0, que es correcto.

    // 3. Convertir el objeto de características a un tensor de TensorFlow.js
    // El orden de las características debe ser EXACTAMENTE el mismo que en X_train de Python.
    // Para obtener el orden correcto en Python, ejecuta X_train.columns.tolist()
    const featureOrder = [
        // Orden alfabético o el orden exacto de X_train.columns.tolist()
        // Este es un ejemplo de orden, DEBES verificar tu X_train.columns.tolist()
        // para asegurarte de que este array sea idéntico.
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', // EmployeeCount no se eliminó en el preprocesamiento,
                                                                             // por lo que debe estar aquí. Asumimos 1.
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
        'Department_Research & Development', 'Department_Sales',
        'JobRole_Human Resources', 'JobRole_Laboratory Technician', 'JobRole_Manager',
        'JobRole_Manufacturing Director', 'JobRole_Research Director', 'JobRole_Research Scientist',
        'JobRole_Sales Executive', 'JobRole_Sales Representative',
        'OverTime_Yes'
    ];

    // Asegurarse de que 'EmployeeCount' y 'EmployeeNumber' estén en el featureOrder si no fueron eliminados
    // y asignarles un valor fijo si no vienen del formulario.
    // Del resumen del dataset original, 'EmployeeCount' siempre es 1 y 'StandardHours' siempre es 80.
    // 'EmployeeNumber' se eliminó en el preprocesamiento de Python al hacer get_dummies y luego drop('Attrition_Yes').
    // Así que, si no lo eliminaste explícitamente, 'EmployeeNumber' también se convirtió a dummies si era categórico,
    // o se mantuvo si era numérico. Lo más seguro es que haya sido numérico y se mantuvo.
    // Si 'EmployeeNumber' y 'EmployeeCount' no fueron explícitamente eliminados en Python, DEBEN estar en el `featureOrder`.
    // Según el PDF, no se eliminaron explícitamente, por lo que podrían estar ahí.
    // Revisemos el X.head() que imprimiste en Colab para el orden de columnas.
    // Basado en el PDF, 'EmployeeCount' debería ser 1 y 'EmployeeNumber' también debería ser tratado como una feature.

    // Vamos a ajustar el featureOrder de forma más robusta.
    // Recomiendo encarecidamente que en Colab, después de `X = df.drop("Attrition_Yes", axis=1)`,
    // imprimas `X.columns.tolist()` para tener la lista exacta y el orden de las columnas.
    // Por ahora, asumiré el orden de la tabla original + las dummies.

    // IMPORTANTE: REEMPLAZA ESTA LISTA CON LA SALIDA REAL DE X.columns.tolist() DE TU COLAB
    const actualFeatureOrderFromPython = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 'Department_Research & Development',
        'Department_Sales', 'EducationField_Life Sciences', 'EducationField_Marketing', 'EducationField_Medical',
        'EducationField_Other', 'EducationField_Technical Degree', 'Gender_Male', 'JobRole_Human Resources',
        'JobRole_Laboratory Technician', 'JobRole_Manager', 'JobRole_Manufacturing Director',
        'JobRole_Research Director', 'JobRole_Research Scientist', 'JobRole_Sales Executive',
        'JobRole_Sales Representative', 'MaritalStatus_Married', 'MaritalStatus_Single',
        'OverTime_Yes' // Esta es la última según tu descripción.
        // Faltaría EmployeeNumber, pero suele eliminarse. Si no lo eliminaste, debería estar aquí.
        // Y la columna 'Over18' también suele eliminarse.
    ];

    // Ahora, construimos el array de valores en el orden correcto
    const inputValues = actualFeatureOrderFromPython.map(col => {
        // Manejar las columnas que NO están en el formulario, asignándoles un valor por defecto o 0
        switch (col) {
            case 'Age': return age;
            case 'MonthlyIncome': return monthlyIncome;
            case 'TotalWorkingYears': return totalWorkingYears;
            case 'YearsAtCompany': return yearsAtCompany;
            case 'EmployeeCount': return 1; // Asumimos que EmployeeCount siempre es 1
            case 'StandardHours': return 80; // Asumimos que StandardHours siempre es 80
            // Para las otras columnas numéricas no en el formulario, las inicializamos a 0
            case 'DailyRate': return 0;
            case 'DistanceFromHome': return 0;
            case 'Education': return 0;
            case 'EnvironmentSatisfaction': return 0;
            case 'HourlyRate': return 0;
            case 'JobInvolvement': return 0;
            case 'JobLevel': return 0;
            case 'JobSatisfaction': return 0;
            case 'MonthlyRate': return 0;
            case 'NumCompaniesWorked': return 0;
            case 'PercentSalaryHike': return 0;
            case 'PerformanceRating': return 0;
            case 'RelationshipSatisfaction': return 0;
            case 'StockOptionLevel': return 0;
            case 'TrainingTimesLastYear': return 0;
            case 'WorkLifeBalance': return 0;
            case 'YearsInCurrentRole': return 0;
            case 'YearsSinceLastPromotion': return 0;
            case 'YearsWithCurrManager': return 0;
            case 'EducationField_Life Sciences': return 0; // Estos son campos dummy que no están en el formulario
            case 'EducationField_Marketing': return 0;
            case 'EducationField_Medical': return 0;
            case 'EducationField_Other': return 0;
            case 'EducationField_Technical Degree': return 0;
            case 'Gender_Male': return 0; // Asumimos un género por defecto o se podría añadir al formulario
            case 'MaritalStatus_Married': return 0;
            case 'MaritalStatus_Single': return 0;

            // Manejar las variables one-hot encoded del formulario
            case 'BusinessTravel_Travel_Frequently': return (businessTravel === 'Travel_Frequently' ? 1 : 0);
            case 'BusinessTravel_Travel_Rarely': return (businessTravel === 'Travel_Rarely' ? 1 : 0);
            case 'Department_Research & Development': return (jobRole.includes('Research') || jobRole.includes('Laboratory') ? 1 : 0); // Esto es una estimación. Idealmente Department debería ser un input.
            case 'Department_Sales': return (jobRole.includes('Sales') ? 1 : 0); // Esto es una estimación.

            case 'JobRole_Human Resources': return (jobRole === 'Human Resources' ? 1 : 0);
            case 'JobRole_Laboratory Technician': return (jobRole === 'Laboratory Technician' ? 1 : 0);
            case 'JobRole_Manager': return (jobRole === 'Manager' ? 1 : 0);
            case 'JobRole_Manufacturing Director': return (jobRole === 'Manufacturing Director' ? 1 : 0);
            case 'JobRole_Research Director': return (jobRole === 'Research Director' ? 1 : 0);
            case 'JobRole_Research Scientist': return (jobRole === 'Research Scientist' ? 1 : 0);
            case 'JobRole_Sales Executive': return (jobRole === 'Sales Executive' ? 1 : 0);
            case 'JobRole_Sales Representative': return (jobRole === 'Sales Representative' ? 1 : 0);
            case 'OverTime_Yes': return (overTime === 'Yes' ? 1 : 0);

            default: return 0; // Para cualquier otra columna no manejada, se establece en 0
        }
    });

    // Crear el tensor de entrada: [1, numero_de_caracteristicas]
    const inputTensor = tf.tensor2d([inputValues], [1, actualFeatureOrderFromPython.length]);

    // console.log("Input Tensor:", inputTensor.dataSync());
    // console.log("Input Tensor Shape:", inputTensor.shape);

    // 4. Realizar la predicción
    const prediction = model.predict(inputTensor);
    const probability = prediction.dataSync()[0]; // Obtener el primer (y único) valor de la predicción

    // 5. Mostrar el resultado
    let message = "";
    let riskClass = "";

    if (probability > 0.5) { // Umbral de 0.5 para clasificar riesgo alto
        message = `Riesgo ALTO de rotación (${(probability * 100).toFixed(2)}%)`;
        riskClass = "high-risk";
    } else {
        message = `Probabilidad BAJA de salida (${(probability * 100).toFixed(2)}%)`;
        riskClass = "low-risk";
    }

    resultDiv.className = ''; // Limpiar clases anteriores
    resultDiv.classList.add(riskClass);
    resultDiv.innerText = message;

    // Limpiar el tensor de la memoria
    tf.dispose(inputTensor);
    tf.dispose(prediction);
});
