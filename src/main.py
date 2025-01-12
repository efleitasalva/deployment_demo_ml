from data.data_loader import load_data
from data.data_processor import process_data
from data.data_splitter import split_data
from model.trainer import train_model
from model.evaluator import evaluate_model
from model.saver import save_model

def main():
    #print("inicio entrenamiento")
    # Cargar los datos
    data = load_data(file_path = "data/raw/Student Depression Dataset.csv")

    # Procesar los datos
    processed_data, target = process_data(
        df=data, 
        numeric_features=data.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        categorical_features=data.select_dtypes(include=['object']).columns.tolist(),
        target_column='Depression'
)
    print(processed_data.head())
    
    
    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(
    data=processed_data.assign(Depression=target),  # Reinserta el objetivo temporalmente
    target_column='Depression'
)
    #print(X_train.shape)
    #print(y_train.shape)
    #print(X_test.shape)
    #print(y_test.shape)

    # Entrenar el modelo
    model = train_model(X_train=X_train, y_train=y_train)
    #print(type(model))

    # Evaluar el modelo
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_data=X_test, y_test=y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"AUC: {auc}")

    # Guardar el modelo
    save_model(model, model_path="models/trained_model")
    
if __name__ == "__main__":
    main()