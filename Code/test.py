from diffusers import StableDiffusionPipeline

# 下载原版 v1-5 到本地
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
)
pipeline.save_pretrained("./stable-diffusion-v1-5-original")