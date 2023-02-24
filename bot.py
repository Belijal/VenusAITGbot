import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL
from PIL import Image
import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, filters, CommandHandler
from io import BytesIO
import random

load_dotenv()

TG_TOKEN = os.getenv('TG_TOKEN') #OUR TELEGRAM TOKEN (XXXXXXXXXXX:XXXXXXXXXXXXXX-XXXXXXXXXXXXX)
MODEL_DATA = os.getenv('MODEL_DATA', 'hassanblend/HassanBlend1.5.1.2') #(PROFILE NAME/MODEL NAME) IN HUGGINGFACE
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = (os.getenv('USE_AUTH_TOKEN', 'true').lower() == 'true')
SAFETY_CHECKER = (os.getenv('SAFETY_CHECKER', 'true').lower() == 'true')
HEIGHT = int(os.getenv('HEIGHT', '512')) #MAKE SURE HEIGHT & WIDHT ARE BOTH MULTIPLES OF 8 
WIDTH = int(os.getenv('WIDTH', '512')) #GOING BELOW 512 IN SOME OF THEM MIGHT RESULT IN LOWER QUAILITY IMAGE
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '50')) #GOOD QUALITY WITH 50 (20 TO TESTING PROMPTS)
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))
BLACKLISTED_WORDS = ["child","Toddler","toddler","Toddlers","toddlers","Kiddie","Kiddies","Family","family","families","Families","kiddie","kiddies","Teeny","teeny","Teenys","teenys","Teenies","Teenies","Child","Childs","Baby", "baby","yeare","oldere","Year","Old", "babypussy","Girl","Girls","Boy","Boys","naked child","naked baby","underage","childtits","child porn","baby porn","underage porn","babys","childs","under18", "young","Young","years","Years","kid","kids","Kid","Kids","under 18","teengirl","teenboy","babyboy","babygirl","newborn","Newborn","Schoolchild","schoolchild","teener","Teener","bambino","Bambino","underaged","pedophile","schoolboy","schoolgirl","Youngster","youngster","youngling","Younglings","juvenile","juveniles","Juvenile","pedo","teen","Teen","girl","boy","Adolf Hitler","adolf hitler","AdolfHitler","1","2","5","6","7","8","9","10","11","12","13","14","15","16","17","18","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","parents","parents",]


#revision = "fp16" if LOW_VRAM_MODE else None
revision = None
torch_dtype = torch.float32 
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

link = "@VenusAi_bsc"
text = "Join our Telegram here."

hyperlink = f'Join our Telegram here {link}'

# load the text2img pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN, vae=vae)
pipe = pipe.to("cuda")

# load the img2img pipeline
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN, vae=vae)
img2imgPipe = img2imgPipe.to("cuda")


# disable safety checker if wanted
def dummy_checker(images, **kwargs): return images, False
if not SAFETY_CHECKER: 
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker


def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio

def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup





def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, photo=None):
   
    
    seed = seed if seed is not None else random.randint(1, 10000)
    generator = torch.cuda.manual_seed_all(seed)

    if photo is not None:
        pipe.to("cuda")
        img2imgPipe.to("cuda")
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((height, width))
        with autocast("cuda"):
            image = img2imgPipe(prompt=[prompt], init_image=init_image,
                                    generator=generator,
                                    
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps)["images"][0]
    else:
        pipe.to("cuda")
        img2imgPipe.to("cuda")
        with autocast("cuda"):
            image = pipe(prompt=[prompt],
                                    generator=generator,
                                    
                                    height=height,
                                    width=width,
                                    guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps)["images"][0]
    return image, seed

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"""
        Welcome {update.effective_user.first_name} to the VenusAI Image Generator Bot!
        Press /help to see all available commands! 
        Have fun and let your sexual imagination run wild!
        """)

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        """
        Here is a list of all available commands! 
        /start - Welcomes you at the hottest Bot on Telegram!
        /help - Shows you all available commands!
        /generate - With this command you can create your desired images!
        """
    )    

async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE, blacklisted_words=BLACKLISTED_WORDS) -> None :
    for word in blacklisted_words:
        if word in update.message.text:
            return await update.message.reply_text("Your prompt contains restricted words. Don't do anything illegal! ", reply_to_message_id=update.message.message_id)
    
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=update.message.text)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_chat.id, image_to_bytes(im), caption=f'"{update.message.text}" (Seed: {seed})' + "\n\n" + hyperlink + "\n\n\n\n*DISCLAIMER: VENUSAI-generated images are the product of an artificial intelligence. The user takes the responsibility to ensure that the generated images do not infringe on any intellectual property rights, privacy rights, any other rights, illegal stuff, harmful, offensive, or inappropriate. The creators of the VENUSAI Bot cannot be held responsible for any misuse! \n\nBeta Version, bugs & overload expected.* " , reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    im, seed = generate_image(prompt=update.message.caption, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_chat.id, image_to_bytes(im), caption=f'"{update.message.caption}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            prompt = replied_message.caption
            im, seed = generate_image(prompt, photo=photo)
        else:
            prompt = replied_message.text
            im, seed = generate_image(prompt)
    elif query.data == "VARIATIONS":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        im, seed = generate_image(prompt, photo=photo)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_chat.id, image_to_bytes(im), caption=f'"{prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)



app = ApplicationBuilder().token(TG_TOKEN).build()

app.add_handler(CommandHandler('generateimg', generate_and_send_photo_from_photo))
app.add_handler(CommandHandler('generate', generate_and_send_photo))
app.add_handler(CommandHandler('start', start))
app.add_handler(CommandHandler('help', help))
app.add_handler(CallbackQueryHandler(button))

app.run_polling()