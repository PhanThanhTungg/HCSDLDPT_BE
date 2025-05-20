import tempfile
import cv2
import cloudinary
import cloudinary.uploader
def upload_image(image, ext, cloud_name='dtaqlb1wi', api_key='767675619893862', api_secret='6nXgb0fCFfTHhOheX2vshpXWqcQ', folder=None):
    print(ext)
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )
  
    upload_params = {}
    if folder:
        upload_params["folder"] = folder
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            cv2.imwrite(tmp.name, image)  # image là ảnh numpy
            tmp.flush()
            result = cloudinary.uploader.upload(tmp.name, **upload_params)
        return result['secure_url']
    except Exception as e:
        raise Exception(f"Lỗi upload: {str(e)}")