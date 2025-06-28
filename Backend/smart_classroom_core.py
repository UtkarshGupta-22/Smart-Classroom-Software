from workingsmartcclassroom import register_student, recognize_and_track

def api_register_student(name: str):
    register_student(name)

def api_start_classroom():
    recognize_and_track()
