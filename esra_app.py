import streamlit as st
from CustModel import Model
import numpy as np
import os
from PIL import Image, ImageDraw
st.write("""
         # ESRA: Emotion sensing and recognition APP
         """
         )
st.write("This is a simple detection model to detect object inside the kids drawing!")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    model = Model.Yolov7(weights= '/content/best_new.pt', conf_thres=0.1)
    give_interpretation(model, file)
 

po = ['Children in the middle of the parent', 'Rainbow', 'Roots of the tree', 'a path leading to the house', 'balloon', 'butterflies', 'cloud', 'eyes', 'family animals', 'flag', 'flowers', 'flying birds', 'food', 'furits', 'grass', 'happy kids', 'house with doors', 'house with windows', 'leaves', 'masjid', 'moon', 'mountains', 'ship', 'smoking chimeny', 'star', 'sun', 'trees with large branches', 'trees with leaves', 'vegitables', 'Water', 'Happy family']
ne =['weapon', 'dead body', 'war/fight', 'sad face', 'big hands', 'big teethes', 'broken family', 'cross eyes', 'dagger', 'fire', 'house with no doors', 'long arms', 'rain', 'red blood', 'sad kids', 'scary animals', 'scary things', 'tree with no leaves']
nu =['Mashroom', 'aeroplane', 'car', 'helicopter', 'sports']
nature_sence = ["Mashroom", "Rainbow", "Roots of the tree", "cloud",
                "flo-", "flowers", "grass", "leaves", "moon", "mountains",
                "rain", "smoking chimeny", "star", "sun", 
                "tree with no leaves","trees with large branches", 
                "trees with leaves"]
food_drinks = ["water", "vegitables", "furits", "food"]

Houses_Buildings = ["a path leading to the house", "house with doors",
                    "house with no doors","house with windows", "masjid"]

Fantasy = ["scary things", "red blood", "flag", "fire", "long arms", 
           "big teethes", "big hands"]
toys_games = ["sports", "ship", "helicopter", "cycle", "car", 
              "balloon", "aeroplane"]
family_home_life = ["Children in the middle of the parent", 
                    "broken family", "broken heart", "happy family",
                    "happy kids", "heart", "sad kids", "eyes", "cross eyes"]

animals_birds = ["scary animals", "scary animals-", "family animals", 
                 "flying birds", "dagger", "butterflies"]


# report 
interpretation_report_v2 = {'balloon': 'Balloons are often associated with celebrations and can evoke feelings of joy and happiness. Children might choose to include balloons in their artwork to express feelings of excitement, anticipation, or happiness.',
 'butterflies': 'Butterflies are often used as symbols of transformation, growth, and freedom. Children might include butterflies in their artwork to express feelings of change, new beginnings, or a desire for freedom.',
 'cloud': 'A cloud can represent many things, including creativity, imagination, and serenity. Children might choose to include clouds in their artwork to express feelings of calmness, peace, or a desire for a peaceful environment.',
 'eyes': 'Eyes can be used to represent insight or understanding. Children might include eyes in their artwork to express a desire for clarity or understanding in a situation, or to convey that they are watching or observing something.',
 'family animals': 'Animals that are often considered pets, such as dogs or cats, can represent warmth, affection, and companionship. Children might include family animals in their artwork to express a desire for connection, love, or companionship.',
 'flag': 'A flag can represent pride, unity, and identity. Children might choose to include a flag in their artwork to express a sense of pride or patriotism, or to explore questions of identity and belonging.',
 'flowers': 'Flowers are often associated with beauty, love, and growth. Children might include flowers in their artwork to express feelings of happiness, love, or a desire for growth and change.',
 'flying birds': 'Flying birds can represent freedom, escape, or a sense of possibility. Children might include flying birds in their artwork to express a desire for freedom or adventure, or to explore ideas of possibility and potential.',
 'food': 'Food is generally positive and can represent sustenance, pleasure, or nourishment. Children might include food in their artwork to express a desire for comfort, pleasure, or to explore their relationship with food.',
 'furits': 'Fruits are generally positive and can represent health, vitality, or sweetness. Children might include fruits in their artwork to express a desire for health and wellness, or to explore the idea of sweetness and pleasure.',
 'grass': 'Grass can represent growth, life, or freshness. Children might include grass in their artwork to express a desire for growth, change, or renewal.',
 'happy kids': 'Happy kids suggest joy, laughter, and a sense of playfulness. Children might choose to include happy kids in their artwork to express feelings of happiness, excitement, or a desire for fun and play.',
 'ship': 'A child who draws a ship might be expressing a desire for adventure or a need to explore new horizons. They may also be feeling a sense of freedom or possibility.',
 'smoking chimney': 'A child who draws a smoking chimney might be expressing a need for warmth, comfort, and a sense of community. They may also be feeling a sense of belonging or a desire for a stable home environment.',
 'star': 'A child who draws a star might be expressing a need for guidance, hope, or inspiration. They may also be feeling a sense of optimism or positivity, or they may be seeking comfort in the night sky.',
 'sun': 'A child who draws a sun might be expressing a need for warmth, light, or the beauty of nature. They may also be feeling a sense of vitality or joy, or they may be seeking comfort in the warmth of the sun.',
 'trees with large branches': 'A child who draws trees with large branches might be expressing a need for strength, shelter, or a connection to nature. They may also be feeling a sense of security or groundedness.',
 'trees with leaves': 'A child who draws trees with leaves might be expressing a need for growth, renewal, or the beauty of nature. They may also be feeling a sense of calmness or tranquility, or they may be seeking comfort in the beauty of the natural world.',
 'vegetables': 'A child who draws vegetables might be expressing a need for health, nourishment, or the goodness of nature. They may also be feeling a sense of well-being or positivity.',
 'water': 'A child who draws water might be expressing a need for life, purification, or the beauty of nature. They may also be feeling a sense of renewal or vitality, or they may be seeking comfort in the soothing properties of water.',
 'happy family': 'A child who draws a happy family might be expressing a need for love, connection, or a sense of belonging. They may also be feeling a sense of warmth or joy, or they may be seeking comfort in the idea of a stable, supportive family.',
 'big hands': 'A child who draws big hands might be expressing a need for power or strength. However, in a negative context, they may be feeling a sense of aggression or domination.',
 'big teeth': 'A child who draws big teeth might be expressing a need for fierceness or strength. However, in a negative context, they may be feeling a sense of threat or danger.',
 'broken family': 'A child who draws a broken family might be expressing sadness, loss, or a sense of being incomplete. They may also be seeking comfort in expressing their emotions through art.',
 'cross eyes': 'A child who draws crossed eyes might be expressing confusion, disorientation, or an inability to focus. They may also be seeking comfort in expressing their emotions through art.',
 'dagger': 'A child who draws a dagger might be expressing a sense of danger, harm, or potential harm. They may also be seeking comfort in expressing their emotions through art.',
 'fire': 'A child who draws fire might be expressing a sense of passion or energy. However, in a negative context, they may be feeling a sense of destruction or danger.',
 'house with no doors': 'A house with no doors suggests a lack of security or access, which could generate feelings of fear or discomfort. This may suggest that the child may be feeling vulnerable or exposed in some way.',
 'long arms': 'Long arms might suggest reach or power, but in a negative context, they might suggest intimidation or threat. This may suggest that the child may feel either empowered or intimidated by someone or something in their life.',
 'rain': 'Rain can represent cleansing or renewal, but in a negative context, it might suggest sadness or gloom. This may suggest that the child may be experiencing feelings of sadness or depression.',
 'red blood': 'Red blood might suggest vitality or life, but in a negative context, it might suggest injury or danger. This may suggest that the child may be feeling threatened or unsafe in some way.',
 'sad kids': 'Sad kids typically suggest a sense of loss or unhappiness. This may suggest that the child may be feeling disconnected or lonely.',
 'scary animals': 'Scary animals might generate fear or danger. This may suggest that the child may be experiencing fear or anxiety.',
 'scary things': 'Scary things might suggest danger, threat, or discomfort. This may suggest that the child may be experiencing some form of trauma or anxiety.',
 'tree with no leaves': 'A tree with no leaves might suggest death, loss, or a lack of vitality. This may suggest that the child may be experiencing feelings of loss or grief.',
 'mushroom': 'A mushroom is neutral and could suggest growth, vitality, or even magical qualities. This may suggest that the child is feeling optimistic and hopeful.',
 'airplane': 'An airplane is neutral and might suggest adventure, travel, or excitement. This may suggest that the child is feeling adventurous or eager for new experiences.',
 'car': 'A car is neutral and might suggest mobility, freedom, or independence. This may suggest that the child is feeling independent and empowered.',
 'helicopter': 'A helicopter is neutral and might suggest adventure, transportation, or emergency response. This may suggest that the child is feeling adventurous or eager to explore.',
 'sports': 'Sports are neutral and might suggest competition, teamwork, or physical activity. This may suggest that the child is feeling competitive or enjoys physical activity.',
 'children in the middle of the parent': 'Children surrounded by their parents might suggest safety, security, and love. This may suggest that the child is feeling loved and supported.',
 'rainbow': 'Rainbows typically suggest hope, positivity, and beauty. This may suggest that the child is feeling optimistic and hopeful.',
 'roots of the tree': 'The roots of a tree may represent stability and groundedness, which could generate feelings of safety or security. This may suggest that the child is feeling stable and secure in their life.',
 'a path leading to the house': 'A path leading to a house might suggest welcoming, safety, and comfort. This may suggest that the child is feeling safe and comforted in their environment.',
 'house with doors': 'A child drawing a house with doors might be expressing a need for safety, security, and control. They may feel vulnerable or anxious and want to protect themselves from external threats.',
 'house with windows': 'A child drawing a house with windows might be expressing a desire for openness, connection, and inspiration. They may feel curious, imaginative, and eager to explore the world around them.',
 'leaves': 'A child drawing leaves might be expressing a sense of growth, renewal, and appreciation for the beauty of nature. They may feel calm, peaceful, and connected to the environment.',
 'masjid': 'A child drawing a masjid might be expressing a sense of spirituality, devotion, and community. They may feel a strong connection to their faith and culture and seek comfort and guidance from their religious practices.',
 'moon': 'A child drawing a moon might be expressing a sense of mystery, imagination, and romance. They may feel curious, dreamy, and fascinated by the unknown.',
 'mountains': 'A child drawing mountains might be expressing a sense of strength, stability, and admiration for the majesty of nature. They may feel inspired, confident, and empowered by the grandeur of the landscape.',
 'weapon': "The interpretation of a child's drawing containing weapons can vary depending on the context and the age of the child. However, it could suggest that the child is feeling threatened, fearful, or is expressing aggression. It could also indicate that the child has been exposed to violent media or surroundings, and may need support and guidance to process these experiences in a healthy way.",
 'dead body': "A child's drawing contains a dead body, it could suggest that the child has been exposed to or experienced some kind of trauma, violence, or loss. It may also indicate feelings of fear, helplessness, or sadness. It is important to approach this situation with sensitivity and seek appropriate professional support to help the child process their emotions and experiences in a healthy way.",
 'sad face': "A sad faces in a child's drawing can indicate feelings of sadness, loneliness, or depression. The child may be experiencing emotional pain or distress, and may benefit from opportunities to express their feelings and receive support from caregivers or mental health professionals.",
 'war/fight': "A child's drawing contains depictions of war or fights, it may suggest that the child is feeling a sense of conflict, aggression, or powerlessness. The child may be expressing feelings of anger, frustration, or a desire for control in a situation that they perceive as threatening or dangerous. It is important to approach such drawings with sensitivity and to explore the child's feelings and experiences in a supportive and non-judgmental way."}
# report   end              

def remove_duplicates(lst):
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": f"Based on the informtion below, combine it into a report that will be useful to parents with a summary of table as well: {txt}"},
    ]
)    

def give_interpretation(model, img_path, kname='Ali', age='10', gender='male', save_output = True):
  result = model.predict(img_path, save_output)
  input_image = Image.open(img_path)
  obj_name = result[0]
  obj_size = result[1]
  conf = result[2]
  unique_obj = set(obj_name)
  # print(unique_obj)
  final_sizes = []
  boxes = []
  #filename = os.path.splitext(os.path.basename(img_path))[0]
  # Create heatmap on the input image
  heatmap_image = Image.new("RGBA", input_image.size, (0, 0, 0, 0))
  draw = ImageDraw.Draw(heatmap_image)
  for bbox, bname in zip(obj_size, obj_name):
      if bname in po:
        draw.rectangle(bbox, fill=(0, 255, 0, 100))
      elif bname in ne:
        draw.rectangle(bbox, fill=(255, 0, 0, 100))
      else:
        draw.rectangle(bbox, fill=(0, 0, 255, 100))

  # Merge input image and heatmap image
  output_image = Image.alpha_composite(input_image.convert("RGBA"), heatmap_image)
  st.image(output_image, use_column_width=True)
  # Save output image
  #output_image.save(f"heatmap_{filename}.png")


  for name, size, con in zip(obj_name, obj_size, conf):
    boxes.append(size)
    fsize = (size[3]-size[1])*(size[2]-size[0])
    final_sizes.append((name, fsize, con))
    
  interp = []  
  for obj in unique_obj:
    total_size = [size for name, size,conf in final_sizes if name==obj]
    total_conf = [float(conf) for name, size, conf in final_sizes if name==obj]
    interp.append((obj, len(total_size), sum(total_size), sum(total_conf)/len(total_conf)))

  sorted_list = sorted(interp, key=lambda x: x[1], reverse=True)
  # print(sorted_list)
  des, emotions, score =[], [], []
  for obj, w, count, con in sorted_list:
    if obj in po:
      score.append(["p", w])
    elif obj in ne:
      score.append(["n", w])
    else:
      score.append(["ne", w])
    if obj in nature_sence:
      des.append("nature sences")
      emotions.append("Calm, peaceful, happy, free")
    elif obj in food_drinks:
      des.append("Food and drinks")
      emotions.append("Hungry, happy, satisfied")
    elif obj in Houses_Buildings:
      des.append("Houses and buildings")
      emotions.append("Safe, cozy, proud")
    elif obj in Fantasy:
      des.append("Fantasy")
      emotions.append("Excited, curious, imaginative")
    elif obj in animals_birds:
      des.append("animals and birds")
      emotions.append("Happy, playful, curious")
    elif obj in family_home_life:
      des.append("home and family related objects")
      emotions.append("Loving, nurturing, protective")
    elif obj in toys_games:
      des.append("Toys and games")
      emotions.append("Playful, happy, creative")
  # print(des)
  des, emotions = remove_duplicates(des), remove_duplicates(emotions)
  p, n, u = 0, 0, 0
  for s in score:
    if s[0]=="p":
      p+=s[-1]
    elif s[0]=="n":
      n+=s[-1]*2
    else:
      u+=s[-1]
  score_list = [p, n, u]
  score_card = np.argmax(score_list)
  
  unique_obj = list(unique_obj)
  # des = ", ".join(des[:-1])+" and "+des[-1]
  # unobj = ", ".join(unique_obj[:-1])+" and "+unique_obj[-1]
  # emotions = ", ".join(emotions[:-1])+" and "+emotions[-1]
  # sorted_list = sorted(interp, key=lambda x: x[1], reverse=True)
  text = f"ESRA app has detected the following objects\n"
  for idx, obj in enumerate(unique_obj):
    confidenc_score = [conf for ob,_, _,conf in interp if ob==obj]
    count = [c for ob,c, _,_ in interp if ob==obj]
    area = [c for ob,_, c,_ in interp if ob==obj]
    count = count[-1]
    area = round((area[0]*100)/(640*640), 2)
    confidenc_score = round(confidenc_score[0])
    if confidenc_score<50:
      val = "reasonably sure"
    elif confidenc_score>50 and confidenc_score<75:
      val = "quite certain"
    elif confidenc_score>75:
      val = "almost certain"
    obj = str.lower(obj)
    text = text + f"Object {idx+1}:\nWe are {val} that there {'are' if count>1 else 'is'} {obj}[s/es] [{confidenc_score}% sure] The object covering {area}% area in the drawing..\nInterpretation: {interpretation_report_v2[obj]} \n\n"
  
  text = text + f"Image Interpretation: \n The Overall drawing conveyed {'negative' if score_card==1 else 'positive'} emotions."
  res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": f"Based on the informtion below, combine it into a report that will be useful to parents with a summary of table as well: {text}"},
    ]
  )
  result = res['choices'][0]['message']['content']
  # print(sorted_list)
  print(f"""
  Kid Information \n
  Name: {kname} \n
  Age: {age}\n
  Gender: {gender}\n
  ****************************************************************************************
  \n
  {result}
  ****************************************************************************************
  \n
  """)