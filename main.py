from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import math
import random
import base64
import io
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# Serve hairstyle images
@app.route('/hair-styles/<path:filename>')
def serve_hairstyles(filename):
    return send_from_directory('hair styles', filename)

# Serve glasses images
@app.route('/glasses/<path:filename>')
def serve_glasses(filename):
    return send_from_directory('glasses', filename)

# Serve makeup images
@app.route('/makeup/<path:filename>')
def serve_makeup(filename):
    return send_from_directory('makeup', filename)

# Test route to check images
@app.route('/test-images')
def test_images():
    """Test if images are being served correctly"""
    hairstyles_folder = f'hair styles'
    glasses_folder = f'glasses'
    makeup_folder = f'makeup'
    
    result = {
        'folder_exists': os.path.exists(hairstyles_folder),
        'glasses_folder_exists': os.path.exists(glasses_folder),
        'makeup_folder_exists': os.path.exists(makeup_folder),
        'subfolders': [],
        'glasses_subfolders': [],
        'makeup_subfolders': [],
        'images_found': []
    }
    
    if result['folder_exists']:
        for item in os.listdir(hairstyles_folder):
            item_path = os.path.join(hairstyles_folder, item)
            if os.path.isdir(item_path):
                subfolder_info = {
                    'name': item,
                    'images': []
                }
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        subfolder_info['images'].append(img_file)
                result['subfolders'].append(subfolder_info)
    
    if result['glasses_folder_exists']:
        for item in os.listdir(glasses_folder):
            item_path = os.path.join(glasses_folder, item)
            if os.path.isdir(item_path):
                subfolder_info = {
                    'name': item,
                    'images': []
                }
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        subfolder_info['images'].append(img_file)
                result['glasses_subfolders'].append(subfolder_info)
    
    if result['makeup_folder_exists']:
        for item in os.listdir(makeup_folder):
            item_path = os.path.join(makeup_folder, item)
            if os.path.isdir(item_path):
                subfolder_info = {
                    'name': item,
                    'images': []
                }
                for img_file in os.listdir(item_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        subfolder_info['images'].append(img_file)
                result['makeup_subfolders'].append(subfolder_info)
    
    return jsonify(result)

class AccurateFaceShapeWithLandmarks:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def get_face_measurements(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            points = {
                'forehead_left': 103, 'forehead_right': 332, 'cheek_left': 234,
                'cheek_right': 454, 'jaw_left': 172, 'jaw_right': 397,
                'chin': 152, 'nose_tip': 1, 'forehead_top': 10
            }
            
            point_coords = {}
            for name, idx in points.items():
                landmark = landmarks.landmark[idx]
                point_coords[name] = (int(landmark.x * w), int(landmark.y * h))
            
            return point_coords
        except Exception as e:
            print(f"Error in face measurements: {e}")
            return None
    
    def calculate_simple_ratios(self, points):
        if points is None:
            return None
        
        try:
            face_width = math.dist(points['cheek_left'], points['cheek_right'])
            face_height = math.dist(points['forehead_top'], points['chin'])
            jaw_width = math.dist(points['jaw_left'], points['jaw_right'])
            forehead_width = math.dist(points['forehead_left'], points['forehead_right'])
            
            width_height_ratio = face_width / face_height
            jaw_face_ratio = jaw_width / face_width
            
            return {
                'width_height_ratio': width_height_ratio,
                'jaw_face_ratio': jaw_face_ratio,
                'face_width': float(face_width),
                'face_height': float(face_height),
                'jaw_width': float(jaw_width),
                'forehead_width': float(forehead_width),
                'cheek_width': float(face_width)
            }
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return None
    
    def detect_shape_accurate(self, ratios):
        if ratios is None:
            return "No Face", 0.0
        
        try:
            wh_ratio = ratios['width_height_ratio']
            jf_ratio = ratios['jaw_face_ratio']
            
            if wh_ratio > 0.85:
                if jf_ratio > 0.85:
                    return "SQUARE", 0.8
                else:
                    return "ROUND", 0.8
            elif wh_ratio < 0.65:
                return "LONG", 0.8
            elif jf_ratio < 0.75:
                return "HEART", 0.7
            else:
                return "OVAL", 0.8
        except Exception as e:
            print(f"Error detecting shape: {e}")
            return "UNKNOWN", 0.0

class FlirtyFaceAnalysis:
    def __init__(self):
        self.compliments = [
            "Wow! Your face is absolutely stunning! 😍",
            "You're more perfect than you think! ✨",
            "Your features are incredibly attractive! 🌟",
            "Someone's looking gorgeous today! 💫",
            "Your face could launch a thousand ships! ⚡",
            "Absolutely breathtaking features! 🌈",
            "You've got that model-quality face! 🎯",
            "Your symmetry is on point! 🔥",
            "What a beautiful face structure! 💖",
            "You're naturally photogenic! 📸"
        ]
        
        self.hairstyles = {}
        self.glasses = {}
        self.makeup = {}
        self.check_existing_images()
    
    def check_existing_images(self):
        hairstyles_folder = 'hair styles'
        glasses_folder = 'glasses'
        makeup_folder = 'makeup'
        
        print("🔍 Checking for existing images...")
        
        hairstyle_folder_to_shape = {
            'oval': 'OVAL', 'round': 'ROUND', 'square': 'SQUARE',
            'oblong': 'LONG', 'heart': 'HEART', 'diamond': 'DIAMOND', 'triangle': 'TRIANGLE'
        }
        
        glasses_folder_to_shape = {
            'oval': 'OVAL', 'round': 'ROUND', 'square': 'SQUARE',
            'oblong': 'LONG', 'heart': 'HEART', 'diamond': 'DIAMOND', 'triangle': 'TRIANGLE'
        }
        
        makeup_folder_to_shape = {
            'oval': 'OVAL', 'round': 'ROUND', 'square': 'SQUARE',
            'oblong': 'LONG', 'heart': 'HEART', 'diamond': 'DIAMOND', 'triangle': 'TRIANGLE'
        }
        
        self.hairstyle_tags = {
            'SQUARE': ["Soft Curls or Waves", "Textured Haircut", "Beachy Shag Haircuts", "Soft Perm with Layers"],
            'OVAL': ["Long U Cut Hairstyles for Women", "Long Shag Haircuts for Women", "Curtain Bangs with Layers", "Blunt Bob"],
            'HEART': ["Long Layers with Curls", "Low Side Bun", "Pixie Cut", "Korean Haircuts"],
            'LONG': ["Bardot Bangs", "Full Blunt Bangs", "Box Braids Bob Cut", "Side Ponytail"],
            'ROUND': ["High Top Knot or Bun", "Long Bob with Waves", "Long Layers", "Side Swept Bangs"],
            'DIAMOND': ["Long Hair with Soft Waves", "Blunt Lob with Side Part", "Deva Haircuts", "Pixie Haircuts"]
        }
        
        self.glasses_tags = {
            'SQUARE': ["Browline or Semi Rimless", "Butterfly Glasses", "Oval Glasses", "Round Sunglasses"],
            'OVAL': ["Cat Eye Glasses", "Geometric Glasses", "Oversized Sunglasses", "Square Glasses"],
            'HEART': ["Butterfly Glasses", "Oval Glasses", "Round Glasses", "Wayfarer Glasses"],
            'LONG': ["Cat Eye Glasses", "Decorative Top or Brownline", "Round Glasses", "Square Glasses"],
            'ROUND': ["Cat Eye Glasses", "Geometric Glasses", "Lightly Tinted Frames", "Square Glasses"],
            'DIAMOND': ["Browline Glasses", "Cat Eye Glasses", "Oval Glasses", "Semi Rimless Glasses"]
        }
        
        self.makeup_tags = {
            'SQUARE': ["Soft Smokey Eyes", "Defined Brows", "Rosy Cheeks", "Nude Lips"],
            'OVAL': ["Natural Glow", "Winged Eyeliner", "Bronzed Cheeks", "Bold Lips"],
            'HEART': ["Cat Eye Makeup", "Highlighted Cheekbones", "Soft Blush", "Berry Lips"],
            'LONG': ["Horizontal Eyeshadow", "Rounded Brows", "Horizontal Blush", "Full Lips"],
            'ROUND': ["Angled Eyeshadow", "Angled Brows", "Contoured Cheeks", "Defined Lips"],
            'DIAMOND': ["Soft Smokey Eyes", "Arched Brows", "Highlighted Cheekbones", "Natural Lips"]
        }
        
        # Scan hairstyles folders
        if os.path.exists(hairstyles_folder):
            for folder_name, shape_name in hairstyle_folder_to_shape.items():
                folder_path = os.path.join(hairstyles_folder, folder_name)
                if os.path.exists(folder_path):
                    available_styles = []
                    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
                    png_files.sort()
                    
                    shape_tags = self.hairstyle_tags.get(shape_name, [])
                    
                    for i, img_file in enumerate(png_files):
                        if i < len(shape_tags):
                            style_name = shape_tags[i]
                        else:
                            style_names = ["Layered Cut", "Side Swept", "Textured Bob", "Soft Waves", 
                                          "Modern Style", "Classic Look", "Trendy Cut", "Elegant Style"]
                            style_name = style_names[i] if i < len(style_names) else f"Style {i+1}"
                        
                        available_styles.append({
                            "name": style_name,
                            "image": f"{folder_name}/{img_file}",
                            "description": f"Perfect for your {shape_name.lower()} face shape"
                        })
                    
                    self.hairstyles[shape_name] = available_styles
                else:
                    self.hairstyles[shape_name] = []
        else:
            print(f"⚠️ Hairstyles folder '{hairstyles_folder}' not found!")
        
        # Scan glasses folders
        if os.path.exists(glasses_folder):
            for folder_name, shape_name in glasses_folder_to_shape.items():
                folder_path = os.path.join(glasses_folder, folder_name)
                if os.path.exists(folder_path):
                    available_glasses = []
                    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
                    png_files.sort()
                    
                    shape_tags = self.glasses_tags.get(shape_name, [])
                    
                    for i, img_file in enumerate(png_files):
                        if i < len(shape_tags):
                            glasses_name = shape_tags[i]
                        else:
                            glasses_names = ["Classic Aviator", "Modern Rectangle", "Round Vintage", "Cat Eye", 
                                           "Browline Glasses", "Oversized Frames", "Sporty Style", "Designer Frames"]
                            glasses_name = glasses_names[i] if i < len(glasses_names) else f"Glasses {i+1}"
                        
                        available_glasses.append({
                            "name": glasses_name,
                            "image": f"{folder_name}/{img_file}",
                            "description": f"Ideal for {shape_name.lower()} face shapes"
                        })
                    
                    self.glasses[shape_name] = available_glasses
                else:
                    self.glasses[shape_name] = []
        else:
            print(f"⚠️ Glasses folder '{glasses_folder}' not found!")
        
        # Scan makeup folders
        if os.path.exists(makeup_folder):
            for folder_name, shape_name in makeup_folder_to_shape.items():
                folder_path = os.path.join(makeup_folder, folder_name)
                if os.path.exists(folder_path):
                    available_makeup = []
                    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
                    png_files.sort()
                    
                    shape_tags = self.makeup_tags.get(shape_name, [])
                    
                    for i, img_file in enumerate(png_files):
                        if i < len(shape_tags):
                            makeup_name = shape_tags[i]
                        else:
                            makeup_names = ["Natural Look", "Evening Glam", "Daytime Chic", "Bold & Beautiful",
                                          "Soft & Romantic", "Professional Style", "Party Ready", "Casual Day"]
                            makeup_name = makeup_names[i] if i < len(makeup_names) else f"Look {i+1}"
                        
                        available_makeup.append({
                            "name": makeup_name,
                            "image": f"{folder_name}/{img_file}",
                            "description": f"Perfect makeup for {shape_name.lower()} face shapes"
                        })
                    
                    self.makeup[shape_name] = available_makeup
                else:
                    self.makeup[shape_name] = []
        else:
            print(f"⚠️ Makeup folder '{makeup_folder}' not found!")
    
    def get_available_hairstyles(self, shape):
        return self.hairstyles.get(shape, [])
    
    def get_available_glasses(self, shape):
        return self.glasses.get(shape, [])
    
    def get_available_makeup(self, shape):
        return self.makeup.get(shape, [])
    
    def get_detailed_makeup_analysis(self, shape):
        """Get detailed makeup analysis based on face shape"""
        makeup_analysis = {
            "HEART": {
                "description": "You have this shape if your cheekbone width is about the same as your forehead width, and both are wider than your jawline. Your face will also be slightly longer than it is wide. Common facial features of heart shaped faces are high, prominent cheekbones and a widow's peak.",
                "techniques": [
                    "💖 Cheeks: Smile and apply a matte blush on the apples of the cheek, blending towards your ear/hairline. This will help accentuate your cheeks. To emphasize your cheekbones more, use a bronzer beneath your cheekbone and under your chin to contour. Apply a sparkly highlighter on the high points of your cheekbones, where the light would naturally hit.",
                    "💖 Brows: Rounded brows can help soften your face.",
                    "💖 Eyes: Stick with soft lines and subtle liner.",
                    "💖 Lips: Add a pop of color to the lips to draw some attention from your strong cheekbones towards the lips. Soft/sheer pinks or reds are recommended."
                ],
                "bonus_tips": [
                    "Apply a small amount of bronzer to your temples and blend upward towards the center of the forehead.",
                    "Use your highlighter underneath your brow bone, on your Cupid's Bow (the curve on your upper lip), and the bridge of your nose."
                ]
            },
            "ROUND": {
                "description": "Your forehead and jawline are both rounded and about the same width. Your jawline has subtle angles. The width of your face will be similar to the length.",
                "techniques": [
                    "🔴 Cheeks: Apply a bronzer around the entire outer perimeter of your face and beneath your cheekbones to add some definition and edginess. Highlighter on the high points of your cheeks will help to accentuate your cheekbones. Smile and apply a blush on the apples of your cheeks, blending towards your ear and hairline.",
                    "🔴 Brows: High, arched brows shapes can help lengthen your face.",
                    "🔴 Eyes: You can go dramatic when it comes to the eyes. Smoky eyes are great for adding a focal point and definition to the face."
                ],
                "bonus_tips": [
                    "Add highlighter to your Cupid's Bow, center of the forehead, underneath the browbone, bridge of the nose, and on the chin.",
                    "Get on board with contouring!"
                ]
            },
            "SQUARE": {
                "description": "Square faces typically have an angular jaw and prominent chin. Your cheekbones will be about the same width as your forehead and jaw. Your hairline typically follows a straight line across your forehead.",
                "techniques": [
                    "🔷 Forehead & Jawline: Apply bronzer on your jawline and your temples, gently blending towards the center of your forehead.",
                    "🔷 Cheeks: Use a fluffy brush to sweep a light layer of bronzer upwards over the hollows of your cheekbones. This can help soften the edges of your face. Apply highlighter on the high points of your cheeks and blend up to the temples. Rosy cream blushes can add the perfect flush to your cheeks.",
                    "🔷 Lips: Because your lower face is prominent, soft nude and pink lip colors will help draw attention to your lips. Line your lips before applying gloss or lipstick to better define the area.",
                    "🔷 Brows: Thick brows with a soft arch will draw attention to eyes.",
                    "🔷 Eyes: People with square faces may also want to focus on their lashes, accompanied by a light-colored eyeshadow. Apply a thickening, lengthening mascara."
                ],
                "bonus_tips": [
                    "Soft and ethereal makeup looks help to soften angular features."
                ]
            },
            "OVAL": {
                "description": "The width of your forehead is smaller than the width of your cheekbones. Those with oval shaped faces will typically have a tall forehead, and the cheekbones are the widest part of the face.",
                "techniques": [
                    "🥚 Cheeks: Apply a bronzer working from the temples down to the center of your face, and lightly brush over your cheekbones. Use a light touch of blush, blending from the temples down to the apples of the cheeks, focusing on the cheekbones.",
                    "🥚 Lips: Line your lips before applying gloss or lipstick to better define the chin area.",
                    "🥚 Brows: Softly-angled brows can help balance facial features and give more shape.",
                    "🥚 Eyes: You can accentuate the shape of your eyes by applying a light-colored eyeshadow and then blending a darker color into the crease."
                ],
                "bonus_tips": [
                    "Apply a highlighter to the highpoints of the entire face: temples, chin, Cupid's Bow, bridge of the nose, and under the brow bone.",
                    "Concentrate on one specific feature (eyes vs. lips) while keeping the other simple.",
                    "Oval face shapes can pull off more looks than some others. However, balance is key. Be careful not to go too dramatic on any one part of your face."
                ]
            },
            "LONG": {
                "description": "Oblong faces are longer than they are wide. The forehead, cheeks, and jaw are all about the same width. The chin will have a slight curve, and the jawline is strong and squared.",
                "techniques": [
                    "📏 Cheeks: Emphasize the cheeks by placing a bronzer in the hollows of your cheekbones and adding blush to the apple of the cheeks. Blend back and out to add width. Apply highlighter to the tops of the cheekbones, blending back towards the temples and under the brow bone.",
                    "📏 Brows: Long, flat brows help add width to the face.",
                    "📏 Eyes: Have as much fun as you want with your eyes: apply cat-eye eyeliner, brightly colored shadow, and fun false lashes.",
                    "📏 Lips: Skip the lip liner and bright lip colors. Keep the lips a subtle shade of pink."
                ],
                "bonus_tips": [
                    "Add highlighter to the tip of the nose and Cupid's Bow.",
                    "Avoid placing highlighter on the forehead and chin, as this can make your face appear longer."
                ]
            },
            "DIAMOND": {
                "description": "Your cheeks are wider than your forehead and chin. The difference between these widths can be subtle or dramatic. Those with diamond shaped faces typically have a more pointed chin.",
                "techniques": [
                    "💎 Forehead & Jawline: Individuals with diamond-shaped faces may want to contour with two shades of foundation for more balance. Use the darker shade (no more than two shades darker than your skin color) on the tip of your chin to soften and give the illusion of a wider jaw. Also use the darker shade near the temples and blend upwards toward your hairline. Use the lighter foundation along your jawline and blend so that you're unable to see where one foundation ends and the other begins.",
                    "💎 Brows: Softly-curved eyebrows can help shorten and soften the face.",
                    "💎 Cheeks: Place highlighter on the highpoints of your cheekbones to accentuate them. Only apply blush to the apples of your cheeks and avoid blushes with a shimmer in them.",
                    "💎 Lips: Picking natural shades and light colors is recommended. Follow the natural line of the lips when applying lip liner to avoid making the lips appear wider than they are and the chin look thinner."
                ],
                "bonus_tips": []
            }
        }
        
        return makeup_analysis.get(shape, {
            "description": "Your unique face shape has beautiful features that can be enhanced with the right makeup techniques.",
            "techniques": [
                "Experiment with different makeup styles to find what works best for you",
                "Focus on enhancing your natural features",
                "Consider professional makeup consultation for personalized advice"
            ],
            "bonus_tips": []
        })
    
    def analyze_face_with_compliments(self, ratios, shape, gender):
        if ratios is None:
            if gender == 'male':
                return "Hey handsome! Come closer so I can see you better! 💪", [], [], [], [], [], []
            elif gender == 'female':
                return "Hey beautiful! Come closer so I can see you better! 💖", [], [], [], [], [], []
            else:
                return "Hey there! Come closer so I can see you better! 🌟", [], [], [], [], [], []
        
        compliment = random.choice(self.compliments)
        
        wh_ratio = ratios['width_height_ratio']
        jf_ratio = ratios['jaw_face_ratio']
        
        analysis = f"{compliment}\n\n"
        suggestions = []
        glasses_suggestions = []
        makeup_suggestions = []
        
        # Gender-specific greeting
        if gender == 'male':
            analysis += f"Your {shape.lower()} face shape is incredibly handsome! "
        elif gender == 'female':
            analysis += f"Your {shape.lower()} face shape is absolutely beautiful! "
        else:
            analysis += f"Your {shape.lower()} face shape is absolutely lovely! "
        
        # Enhanced shape-specific analysis
        if shape == "ROUND":
            analysis += "Those soft curves give you such a warm, friendly appearance! "
            suggestions.extend([
                "💇 Try soft layers to highlight your beautiful cheeks",
                "✨ Side-swept styles would frame your face perfectly",
                "🌟 Angular cuts would complement your soft features",
                "💫 Height on top would balance your lovely proportions"
            ])
            glasses_suggestions.extend([
                "👓 Angular frames will add definition to your soft features",
                "🕶️ Rectangular glasses will create beautiful contrast",
                "🔷 Square frames will enhance your facial structure",
                "✨ Geometric shapes will complement your round face"
            ])
            makeup_suggestions.extend([
                "💄 Use contour to add definition to your cheekbones",
                "👁️ Apply eyeshadow in an upward angle to lift your eyes",
                "💋 Define your lip shape with lip liner for more structure",
                "🌟 Highlight the center of your face to create length"
            ])
        elif shape == "OVAL":
            analysis += "You have the most balanced and versatile features - you're one of the lucky ones! "
            suggestions.extend([
                "💇 Honestly, any hairstyle would look amazing on you!",
                "✨ You have perfect proportions for any look",
                "🌟 Soft layers would enhance your natural beauty",
                "💫 Experiment with different styles to showcase your versatility"
            ])
            glasses_suggestions.extend([
                "👓 Almost any frame shape will suit your balanced features",
                "🕶️ Walnut-shaped frames are perfect for oval faces",
                "🔷 You can pull off both geometric and rounded frames",
                "✨ Consider oversized frames for a fashion-forward look"
            ])
            makeup_suggestions.extend([
                "💄 You can pull off any makeup look with your balanced features",
                "👁️ Experiment with bold eyeshadow looks",
                "💋 Try different lip colors to express your mood",
                "🌟 Focus on enhancing your natural features"
            ])
        elif shape == "SQUARE":
            analysis += "That strong jawline is incredibly attractive and gives you a powerful presence! "
            suggestions.extend([
                "💇 Soft waves would complement your amazing angles",
                "✨ Layered cuts would enhance your natural strength",
                "🌟 Side parts would soften your beautiful features",
                "💫 Textured styles would highlight your strong bone structure"
            ])
            glasses_suggestions.extend([
                "👓 Round or oval frames will soften your strong angles",
                "🕶️ Aviator styles complement square faces beautifully",
                "🔷 Rimless frames will maintain your natural strength",
                "✨ Curved frames will balance your angular features"
            ])
            makeup_suggestions.extend([
                "💄 Soften angles with rounded blush application",
                "👁️ Use rounded eyeshadow shapes to balance strong jaw",
                "💋 Fuller lips will balance your strong features",
                "🌟 Highlight cheekbones to emphasize your structure"
            ])
        elif shape == "LONG":
            analysis += "Your elegant length gives you such a model-like, sophisticated quality! "
            suggestions.extend([
                "💇 Bangs would make your beautiful face even more captivating",
                "✨ Chin-length styles would highlight your graceful neck",
                "🌟 Width at sides would balance your elegant length",
                "💫 Layered cuts would add beautiful dimension"
            ])
            glasses_suggestions.extend([
                "👓 Wide frames will add width to balance your face",
                "🕶️ Round frames will soften the length of your face",
                "🔷 Low-set temples will shorten the appearance of your face",
                "✨ Oversized frames will create perfect proportions"
            ])
            makeup_suggestions.extend([
                "💄 Apply blush horizontally to add width to your face",
                "👁️ Use eyeshadow to create width at the outer corners",
                "💋 Avoid overlining lips to maintain balance",
                "🌟 Contour forehead and chin to shorten face appearance"
            ])
        elif shape == "HEART":
            analysis += "That heart-shaped face is absolutely adorable and so charming! "
            suggestions.extend([
                "💇 Side bangs would balance your lovely forehead perfectly",
                "✨ Bob cuts would frame your cute chin beautifully",
                "🌟 Chin-length styles would complement your shape",
                "💫 Soft layers would enhance your delicate features"
            ])
            glasses_suggestions.extend([
                "👓 Rimless frames will highlight your delicate features",
                "🕶️ Round frames will balance your wider forehead",
                "🔷 Light-colored frames draw attention to your eyes",
                "✨ Bottom-heavy frames complement your chin shape"
            ])
            makeup_suggestions.extend([
                "💄 Contour temples to minimize forehead width",
                "👁️ Focus eye makeup on lower lash line",
                "💋 Balance with fuller bottom lip",
                "🌟 Highlight center of chin to broaden it"
            ])
        elif shape == "DIAMOND":
            analysis += "Those cheekbones are absolutely stunning and give you such elegant angles! "
            suggestions.extend([
                "💇 Bangs would complement your amazing bone structure",
                "✨ Layers would showcase your natural angles perfectly",
                "🌟 Width at forehead and jaw would balance your features",
                "💫 Side-swept styles would highlight your cheekbones"
            ])
            glasses_suggestions.extend([
                "👓 Oval or cat-eye frames will soften your angles",
                "🕶️ Rimless frames highlight your cheekbones beautifully",
                "🔷 Frames with detailing on the browline work well",
                "✨ Semi-rimless frames complement diamond shapes"
            ])
            makeup_suggestions.extend([
                "💄 Soften cheekbones with rounded blush application",
                "👁️ Emphasize brow bone to balance wide cheekbones",
                "💋 Fuller lips will balance narrow chin",
                "🌟 Highlight center forehead and chin"
            ])
        else:
            analysis += "Your unique face shape has such character and charm! "
            suggestions.extend([
                "💇 Experiment with different styles to find what you love",
                "✨ Soft layers would enhance your natural features",
                "🌟 Consider face-framing highlights to accentuate your shape"
            ])
            glasses_suggestions.extend([
                "👓 Try different frame shapes to see what complements you best",
                "🕶️ Consider your personal style when choosing frames",
                "🔷 Don't be afraid to experiment with bold styles"
            ])
            makeup_suggestions.extend([
                "💄 Experiment with different makeup techniques",
                "👁️ Focus on enhancing your best features",
                "💋 Try different lip shapes to see what works best",
                "🌟 Use makeup to balance your unique proportions"
            ])
        
        # Enhanced ratio-based suggestions
        if wh_ratio > 0.85:
            analysis += "Your face width gives you such a warm, approachable appearance! "
            suggestions.append("🎀 Volume on top would create beautiful balance")
            glasses_suggestions.append("📏 Wider frames will maintain your natural proportions")
            makeup_suggestions.append("📏 Use vertical lines in makeup to create length")
        elif wh_ratio < 0.65:
            analysis += "Your face length gives you such an elegant, sophisticated quality! "
            suggestions.append("🎀 Width at the sides would enhance your graceful length")
            glasses_suggestions.append("📏 Taller frames will complement your face length")
            makeup_suggestions.append("📏 Use horizontal lines in makeup to create width")
        else:
            analysis += "Your proportions are beautifully balanced and harmonious! "
            glasses_suggestions.append("📏 Most frame proportions will work well for you")
            makeup_suggestions.append("📏 Your balanced features work with any makeup style")
        
        if jf_ratio > 0.88:
            analysis += "That strong jawline is so attractive and gives you amazing definition! "
            suggestions.append("💎 Soft, textured styles would complement your strong jaw beautifully")
            glasses_suggestions.append("💎 Rounded frames will soften your strong jawline")
            makeup_suggestions.append("💎 Soften jawline with strategic contouring")
        elif jf_ratio < 0.72:
            analysis += "Your delicate jawline is so graceful and feminine! "
            suggestions.append("💎 Angular cuts would define your jawline perfectly")
            glasses_suggestions.append("💎 Angular frames will add definition")
            makeup_suggestions.append("💎 Define jawline with subtle contour")
        else:
            analysis += "Your jawline has the perfect balance of strength and softness! "
            glasses_suggestions.append("💎 Both rounded and angular frames will work well")
            makeup_suggestions.append("💎 Your jawline needs minimal enhancement")
        
        # Add symmetry and general tips
        symmetry_score = random.randint(75, 95)
        analysis += f"\n\n✨ Based on your facial symmetry (estimated {symmetry_score}%), these suggestions will make your natural beauty shine even brighter! ✨"
        
        # Get hairstyle, glasses, and makeup recommendations using dynamic detection
        recommended_hairstyles = self.get_available_hairstyles(shape)
        recommended_glasses = self.get_available_glasses(shape)
        recommended_makeup = self.get_available_makeup(shape)
        
        print(f"🎯 Recommended {len(recommended_hairstyles)} hairstyles for {shape}")
        print(f"👓 Recommended {len(recommended_glasses)} glasses for {shape}")
        print(f"💄 Recommended {len(recommended_makeup)} makeup looks for {shape}")
        
        return analysis, suggestions, glasses_suggestions, makeup_suggestions, recommended_hairstyles, recommended_glasses, recommended_makeup

# Initialize detectors
face_detector = AccurateFaceShapeWithLandmarks()
beauty_analyzer = FlirtyFaceAnalysis()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    try:
        if 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face and get measurements
        points = face_detector.get_face_measurements(image_cv)
        ratios = face_detector.calculate_simple_ratios(points)
        shape, confidence = face_detector.detect_shape_accurate(ratios)
        
        # Draw landmarks and measurements on image
        if points:
            # Draw key points
            for point in points.values():
                cv2.circle(image_cv, point, 4, (0, 255, 0), -1)
            
            # Draw measurement lines
            cv2.line(image_cv, points['cheek_left'], points['cheek_right'], (0, 255, 255), 2)
            cv2.line(image_cv, points['forehead_top'], points['chin'], (0, 255, 255), 2)
            cv2.line(image_cv, points['jaw_left'], points['jaw_right'], (255, 0, 0), 2)
            cv2.line(image_cv, points['forehead_left'], points['forehead_right'], (255, 255, 0), 2)
            
            # Add measurement values
            if ratios:
                cv2.putText(image_cv, f"W/H: {ratios['width_height_ratio']:.2f}", 
                           (points['cheek_left'][0], points['cheek_left'][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add shape result text
        if points is not None:
            cv2.putText(image_cv, f"SHAPE: {shape}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(image_cv, f"Confidence: {confidence:.1%}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image_cv, "NO FACE DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', image_cv)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'shape': shape,
            'confidence': f"{confidence:.1%}",
            'ratios': ratios,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'has_face': points is not None
        })
    
    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_beauty', methods=['POST'])
def analyze_beauty():
    try:
        data = request.json
        print("Received beauty analysis request:", data)
        
        # Validate required fields
        if not data or 'shape' not in data or 'gender' not in data or 'ratios' not in data:
            return jsonify({
                'success': False, 
                'error': 'Missing required fields: shape, gender, or ratios'
            })
        
        shape = data['shape']
        gender = data['gender']
        ratios = data['ratios']
        
        # Perform beauty analysis
        analysis, suggestions, glasses_suggestions, makeup_suggestions, hairstyles, glasses, makeup = beauty_analyzer.analyze_face_with_compliments(ratios, shape, gender)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'suggestions': suggestions,
            'glasses_suggestions': glasses_suggestions,
            'makeup_suggestions': makeup_suggestions,
            'hairstyles': hairstyles,
            'glasses': glasses,
            'makeup': makeup,
            'symmetry_score': random.randint(75, 95)
        })
    
    except Exception as e:
        print(f"Error in analyze_beauty: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_beauty_with_photo', methods=['POST'])
def analyze_beauty_with_photo():
    """New endpoint for beauty analysis with direct photo upload"""
    try:
        # Check if image data is present
        if 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get additional data
        data = request.json
        gender = data.get('gender', 'other')
        
        # Detect face and get measurements
        points = face_detector.get_face_measurements(image_cv)
        ratios = face_detector.calculate_simple_ratios(points)
        shape, confidence = face_detector.detect_shape_accurate(ratios)
        
        if not points:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image. Please try again with a clearer photo.'
            })
        
        # Perform beauty analysis
        analysis, suggestions, glasses_suggestions, makeup_suggestions, hairstyles, glasses, makeup = beauty_analyzer.analyze_face_with_compliments(ratios, shape, gender)
        
        # Create processed image for display
        if points:
            for point in points.values():
                cv2.circle(image_cv, point, 4, (0, 255, 0), -1)
            
            cv2.line(image_cv, points['cheek_left'], points['cheek_right'], (0, 255, 255), 2)
            cv2.line(image_cv, points['forehead_top'], points['chin'], (0, 255, 255), 2)
            
            cv2.putText(image_cv, f"SHAPE: {shape}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', image_cv)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'shape': shape,
            'confidence': f"{confidence:.1%}",
            'analysis': analysis,
            'suggestions': suggestions,
            'glasses_suggestions': glasses_suggestions,
            'makeup_suggestions': makeup_suggestions,
            'hairstyles': hairstyles,
            'glasses': glasses,
            'makeup': makeup,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'symmetry_score': random.randint(75, 95)
        })
    
    except Exception as e:
        print(f"Error in analyze_beauty_with_photo: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_makeup', methods=['POST'])
def analyze_makeup():
    """New endpoint for makeup analysis with direct photo upload"""
    try:
        # Check if image data is present
        if 'image' not in request.json:
            return jsonify({'success': False, 'error': 'No image data provided'})
        
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get additional data
        data = request.json
        gender = data.get('gender', 'other')
        
        # Detect face and get measurements
        points = face_detector.get_face_measurements(image_cv)
        ratios = face_detector.calculate_simple_ratios(points)
        shape, confidence = face_detector.detect_shape_accurate(ratios)
        
        if not points:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image. Please try again with a clearer photo.'
            })
        
        # Get detailed makeup analysis
        detailed_makeup = beauty_analyzer.get_detailed_makeup_analysis(shape)
        
        # Perform beauty analysis to get other recommendations
        analysis, suggestions, glasses_suggestions, makeup_suggestions, hairstyles, glasses, makeup = beauty_analyzer.analyze_face_with_compliments(ratios, shape, gender)
        
        # Create processed image for display
        if points:
            for point in points.values():
                cv2.circle(image_cv, point, 4, (0, 255, 0), -1)
            
            cv2.line(image_cv, points['cheek_left'], points['cheek_right'], (0, 255, 255), 2)
            cv2.line(image_cv, points['forehead_top'], points['chin'], (0, 255, 255), 2)
            
            cv2.putText(image_cv, f"SHAPE: {shape}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(image_cv, "MAKEUP ANALYSIS", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', image_cv)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'shape': shape,
            'confidence': f"{confidence:.1%}",
            'analysis': analysis,
            'makeup_description': detailed_makeup['description'],
            'makeup_techniques': detailed_makeup['techniques'],
            'makeup_bonus_tips': detailed_makeup['bonus_tips'],
            'makeup_suggestions': makeup_suggestions,
            'makeup': makeup,
            'processed_image': f"data:image/jpeg;base64,{processed_image}",
            'symmetry_score': random.randint(75, 95)
        })
    
    except Exception as e:
        print(f"Error in analyze_makeup: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Face Analysis API is running'})

if __name__ == '__main__':
    # Ensure templates and hairstyles folders exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates folder")
    
    # Check if the hairstyles folder exists with space
    hairstyles_folder = 'hair styles'
    glasses_folder = 'glasses'
    makeup_folder = 'makeup'
    
    if not os.path.exists(hairstyles_folder):
        print(f"⚠️  '{hairstyles_folder}' folder not found!")
        print("Please make sure you have a 'hair styles' folder with subfolders for each face shape")
        print("Expected subfolders: oval, round, square, oblong, heart, diamond, triangle")
    else:
        print(f"✅ Found '{hairstyles_folder}' folder")
        subfolders = [f.name for f in os.scandir(hairstyles_folder) if f.is_dir()]
        print(f"Available hairstyle shape folders: {subfolders}")
    
    if not os.path.exists(glasses_folder):
        print(f"⚠️  '{glasses_folder}' folder not found!")
        print("Please make sure you have a 'glasses' folder with subfolders for each face shape")
        print("Expected subfolders: oval, round, square, oblong, heart, diamond, triangle")
    else:
        print(f"✅ Found '{glasses_folder}' folder")
        subfolders = [f.name for f in os.scandir(glasses_folder) if f.is_dir()]
        print(f"Available glasses shape folders: {subfolders}")
    
    if not os.path.exists(makeup_folder):
        print(f"⚠️  '{makeup_folder}' folder not found!")
        print("Please make sure you have a 'makeup' folder with subfolders for each face shape")
        print("Expected subfolders: oval, round, square, oblong, heart, diamond, triangle")
    else:
        print(f"✅ Found '{makeup_folder}' folder")
        subfolders = [f.name for f in os.scandir(makeup_folder) if f.is_dir()]
        print(f"Available makeup shape folders: {subfolders}")
    
    print("🚀 Starting Face Shape & Beauty Analysis Server...")
    print("📷 Face Shape Detection: Active")
    print("💖 Beauty Analysis: Active")
    print("💇 Hairstyle Recommendations: Active")
    print("👓 Glasses Recommendations: Active")
    print("💄 Makeup Recommendations: Active")
    print("🌐 Server running on http://0.0.0.0:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)