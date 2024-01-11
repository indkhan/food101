import gradio as gr
from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model(
    "hope2")


food_categories = ["Apple pie", "Baby back ribs", "Baklava", "Beef carpaccio", "Beef tartare", "Beignets", "Bibimbap", "Bread pudding", "Breakfast burrito", "Bruschetta", "Caesar salad", "Cannoli", "Caprese salad", "Carrot cake", "Ceviche", "Cheese plate", "Cheesecake", "Chicken curry", "Chicken quesadilla", "Chicken wings", "Chocolate cake", "Chocolate mousse", "Churros", "Clam chowder", "Club sandwich", "Crab cakes", "Creme brulee", "Croque madame", "Cup cakes", "Deviled eggs", "Donuts", "Dumplings", "Edamame", "Eggs benedict", "Escargots", "Falafel", "Filet mignon", "Fish and chips", "Foie gras", "French fries", "French onion soup", "French toast", "Fried calamari", "Fried rice", "Frozen yogurt", "Garlic bread", "Gnocchi", "Greek salad",
                   "Grilled cheese sandwich", "Grilled salmon", "Guacamole", "Gyoza", "Hamburger", "Hot and sour soup", "Hot dog", "Huevos rancheros", "Hummus", "Ice cream", "Lasagna", "Lobster bisque", "Lobster roll sandwich", "Macaroni and cheese", "Macarons", "Margarita", "Miso soup", "Mussels", "Nachos", "Omelette", "Onion rings", "Oysters", "Pad thai", "Paella", "Pancakes", "Panna cotta", "Peking duck", "Pho", "Pizza", "Pork chop", "Poutine", "Prime rib", "Pulled pork sandwich", "Ramen", "Ravioli", "Red velvet cake", "Risotto", "Sashimi", "Scallops", "Seaweed salad", "Shrimp and grits", "Shrimp scampi", "Spaghetti bolognese", "Spaghetti carbonara", "Spring rolls", "Steak", "Strawberry shortcake", "Sushi", "Tacos", "Takoyaki", "Tiramisu", "Tuna tartare", "Waffles"]


def food_category(img):

    image_data = tf.io.read_file(img)
    image = tf.image.decode_image(image_data)
    img_shape = 224
    image = tf.image.resize(image, [img_shape, img_shape])

    image = image[:, :, :3]

    image = tf.expand_dims(image, axis=0)
    # food101hope.save("/content/drive/Othercomputers/My Laptop/code/MacDeepL/food101/finaladvance07_efficientnetb0_fine_tuned_101_classes_augmentation/")

    pred = model.predict(image)

    pred_class_int = int(pred.argmax(axis=1)[0])
    pred_name = food_categories[pred_class_int]
    return (pred_name)


def gradiapp(img):
    return food_category(img)


# Create the interface
interface = gr.Interface(fn=gradiapp, inputs="image", outputs="text")

# Launch the interface
interface.launch()
