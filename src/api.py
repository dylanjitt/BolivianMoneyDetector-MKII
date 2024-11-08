from fastapi import FastAPI, Depends, Query,UploadFile, File, HTTPException, status, Depends
#from llama_index.core.agent import ReActAgent
from src.config import get_settings
from src.detector import BilleteDetector,LLM#, gen_oudia
from functools import cache
from fastapi.responses import Response,JSONResponse,FileResponse
import io,os
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_bill_detector() -> BilleteDetector:
    print("Creating model...")
    return BilleteDetector()

@cache
def get_llm()-> LLM:
  print('getting LLM')
  return LLM()

@app.post("/detectMoney")
def detect_money(
  threshold: float = 0.5,
  file: UploadFile = File(...),
  # n:float=0.0,
  detector: BilleteDetector=Depends(get_bill_detector)
)->Response:
  img_stream = io.BytesIO(file.file.read())
  if file.content_type.split("/")[0] != "image":
      raise HTTPException(
          status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
      )

  try:
      img_obj = Image.open(img_stream)
  except UnidentifiedImageError:
      raise HTTPException(
          status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
      )

  try:
        # Open the image with PIL and convert to numpy array
        img_obj = Image.open(img_stream)
        img_np = np.array(img_obj)
        
        # Convert to BGR format for OpenCV compatibility
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
  except UnidentifiedImageError:
      raise HTTPException(
          status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not supported"
      )
  total,img_det=detector.showImg(img_bgr)

  img_pil = Image.fromarray(img_det)
  image_stream = io.BytesIO()
  img_pil.save(image_stream, format="JPEG")
  image_stream.seek(0)
  return Response(content=image_stream.read(), media_type="image/jpeg")

@app.get("/summaryBiletes")
def info_billetes(
    spanish: bool = False,
    voice: bool = False,
    llm: LLM = Depends(get_llm)
) -> JSONResponse:
    # Generate the description and audio path (if requested)
    generated_text, audio_path = llm.generate_response(spanish=spanish, voice=voice)
    
    # Create the response structure
    response_data = {"description": generated_text}
    
    if voice and audio_path:
        # Include the URL to access the audio file
        response_data["audio_url"] = f"/audio/{os.path.basename(audio_path)}"
    
    return JSONResponse(content=response_data)
    
@app.get("/audio/{filename}")
def get_audio_file(filename: str='audio.wav'):
    file_path = os.path.join("audio", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")









if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")