from facenet_pytorch import InceptionResnetV1

def fixed_image_standardization(image_tensor):
    processed_tensor = image_tensor * 2 - 1
    return processed_tensor

class FaceNet(InceptionResnetV1):
    def forward(self, x):
        x = fixed_image_standardization(x)
        return super().forward(x)
