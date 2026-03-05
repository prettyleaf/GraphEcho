# from clearml import Task
from ultralytics import YOLO


if __name__ == '__main__':
    model_variant = "yolo11l-seg"
    epochs = 100
    dataset_yaml = "../export_points_lv_800x600/dataset.yaml"
    model = YOLO(f'{model_variant}.pt')
    task_name = f'{model_variant}_100_l'

    # task = Task.init(
    #     project_name='Profilometer',
    #     task_name=task_name, )
    # task.set_parameter(name="model_variant", value=model_variant)

    args = dict(
        patience=10,
        data=dataset_yaml,
        epochs=epochs,
        imgsz=800,
        batch=20,
        mosaic=0.0,
        project='./runs',
        name=task_name,
        exist_ok=True,
        device=[0, 1]
    )

    # task.connect(args)
    results = model.train(**args)
