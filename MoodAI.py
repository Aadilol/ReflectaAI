from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from google import genai
from collections import Counter
import random
import re

# === Gemini Client Setup ===
api_key = "AIzaSyCfQF-IORV8C6NH_k0FcYUyuicsTXH5eUg"
client = genai.Client(api_key=api_key)

# === Emotion Detection Model ===
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None  # replaces return_all_scores=True
)

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

emotion_recommendations_pool = {
    "joy": {
        "books": [
            "*The House in the Cerulean Sea* (https://www.goodreads.com/book/show/45047384-the-house-in-the-cerulean-sea?from_search=true&from_srp=true&qid=zGBhFYL53A&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1569514209i/45047384.jpg",
            "*Anxious People* (https://www.goodreads.com/book/show/49127718-anxious-people?ref=nav_sb_ss_1_13) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1597575031i/49127718.jpg",
            "*Big Magic: Creative Living Beyond Fear* (https://www.goodreads.com/book/show/24453082-big-magic?from_search=true&from_srp=true&qid=dS6xj8rE4g&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1451446242i/24453082.jpg",
            "*The Little Paris Bookshop* (https://www.goodreads.com/book/show/23278537-the-little-paris-bookshop?ref=nav_sb_ss_1_25) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1412462018i/23278537.jpg",
            "*84, Charing Cross Road* (https://www.goodreads.com/book/show/368916.84_Charing_Cross_Road?ref=nav_sb_ss_1_22) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1662637640i/368916.jpg",
        ],
        "videos": [
            "[Park and Rec but it's just Pawnee having the WORST companies](https://www.youtube.com/watch?v=AS6Cvb1vZW8&list=WL&index=8)",
            "[Office COLD OPENS To Watch In The Morning](https://www.youtube.com/watch?v=4zyINn5yb4Y)",
            "[Brooklyn 99 moments that make me wish I was part of the 99](https://www.youtube.com/watch?v=mIpvd2Cf1l4)",
            "[Community having world-class writing for 26 minutes straight](https://www.youtube.com/watch?v=efbSijkAwKI)",
            "[Teletubbies Reunion ](https://www.youtube.com/watch?v=jG41tYE-A-4)"
        ],
        "music": [
            "[Feel-Good Hits](https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC)",
            "[Happy Hits](https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC)",
            "[Mood Booster](https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0)",
            "[Throwback Party](https://open.spotify.com/playlist/37i9dQZF1DWYmmr74INQlb)",
            "[Dance Party](https://open.spotify.com/playlist/37i9dQZF1DX0BcQWzuB7ZO)"
        ],
        "tv": [
            "*Bluey* (https://www.imdb.com/title/tt7678620/?ref_=chttvtp_t_14) | https://m.media-amazon.com/images/M/MV5BYWU1YmQzMjEtMDNjOS00MGIyLWExY2ItZDAzNmU5NWViMGZmXkEyXkFqcGc@._V1_.jpg",
            "*Cowboy Bebop* (https://www.imdb.com/title/tt0213338/?ref_=chttvtp_t_44) | https://m.media-amazon.com/images/M/MV5BMTU3ZTdiOGQtYmYwYy00OGM5LThmNjMtZGJmNTVlZjk1ZmEyXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Friends* (https://www.imdb.com/title/tt0108778/?ref_=chttvtp_t_54) | https://m.media-amazon.com/images/M/MV5BOTU2YmM5ZjctOGVlMC00YTczLTljM2MtYjhlNGI5YWMyZjFkXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Blackadder Goes Forth* (https://www.imdb.com/title/tt0096548/?ref_=chttvtp_t_70) | https://m.media-amazon.com/images/M/MV5BM2ZiODg3ZWQtMzcyMC00MTRhLWI2MjItNjk5OTdlOTRiMGRiXkEyXkFqcGc@._V1_.jpg",
            "*Freaks and Geeks* (https://www.imdb.com/title/tt0193676/?ref_=chttvtp_t_73) | https://m.media-amazon.com/images/M/MV5BNDk4MTRlZjMtYTMxYi00ZjdkLWEyYjYtZjg1NjBlMzY2MDIzXkEyXkFqcGc@._V1_.jpg",
        ]
    },
    "sadness": {
        "books": [
            "📖 *The Comfort Book* by Matt Haig",
            "📖 *Reasons to Stay Alive* by Matt Haig",
            "📖 *It's OK That You're Not OK* by Megan Devine",
            "📖 *Lost Connections* by Johann Hari",
            "📖 *The Art of Happiness* by Dalai Lama",
        ],
        "videos": [
            "🎥 [This is What Depression Feels Like - TED](https://youtu.be/XiCrniLQGYc)",
            "🎥 [How to Get Out of a Rut](https://youtu.be/ELpfYCZa87g)",
            "🎥 [The Power of Vulnerability](https://youtu.be/iCvmsMzlF7o)",
            "🎥 [The Gift of Sadness](https://youtu.be/N7o2UjvRdxI)",
            "🎥 [Why We All Need Emotional First Aid](https://youtu.be/1Evwgu369Jw)"
        ],
        "music": [
            "🎧 [Rainy Day Comfort](https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0)",
            "🎧 [Soft Pop Hits](https://open.spotify.com/playlist/37i9dQZF1DWZtZ8vUCzche)",
            "🎧 [Sad Indie](https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1)",
            "🎧 [Deep Focus](https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ)",
            "🎧 [Peaceful Piano](https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO)"
        ],
        "tv": [
            "🎙️ *Terrible, Thanks for Asking*",
            "🎙️ *Unlocking Us* by Brené Brown",
            "🎙️ *Therapy Chat*",
            "🎙️ *On Being*",
            "🎙️ *The Mindful Kind*"
        ]
    },
    "anger": {
        "books": [
            "📖 *Anger: Wisdom for Cooling the Flames* by Thich Nhat Hanh",
            "📖 *Radical Acceptance* by Tara Brach",
            "📖 *Emotional Intelligence* by Daniel Goleman",
            "📖 *Nonviolent Communication* by Marshall Rosenberg",
            "📖 *Letting Go* by David R. Hawkins",
        ],
        "videos": [
            "🎥 [How to Make Stress Your Friend - TED](https://youtu.be/RcGyVTAoXEU)",
            "🎥 [The Power of Letting Go](https://youtu.be/Ji7B3Fky6MA)",
            "🎥 [Getting Rid of Anger](https://youtu.be/zCmhZ6I6jFY)",
            "🎥 [How to Stay Calm](https://youtu.be/TXfDPnNS15A)",
            "🎥 [Managing Emotions Mindfully](https://youtu.be/vzKryaN44ss)"
        ],
        "music": [
            "🎧 [Lo-Fi Chillhop](https://open.spotify.com/playlist/37i9dQZF1DWZ3xRUWZ2ZrL)",
            "🎧 [Chill Vibes](https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6)",
            "🎧 [Instrumental Chill](https://open.spotify.com/playlist/37i9dQZF1DX4E3UdUs7fUx)",
            "🎧 [Cool Down](https://open.spotify.com/playlist/37i9dQZF1DWSqBruwoIXkA)",
            "🎧 [Soothing Strings](https://open.spotify.com/playlist/37i9dQZF1DX9uKNf5jGX6m)"
        ],
        "tv": [
            "🎙️ *The Calm Collective*",
            "🎙️ *The Daily Meditation Podcast*",
            "🎙️ *The Mindful Minute*",
            "🎙️ *10% Happier with Dan Harris*",
            "🎙️ *Let’s Talk About Mental Health*"
        ]
    },
    "fear": {
        "books": [
            "📖 *Feel the Fear and Do It Anyway* by Susan Jeffers",
            "📖 *The Gifts of Imperfection* by Brené Brown",
            "📖 *Dare to Lead* by Brené Brown",
            "📖 *Anxiety Relief* by Judson Brewer",
            "📖 *The Anxiety Toolkit* by Alice Boyes",
        ],
        "videos": [
            "🎥 [The Gift and Power of Emotional Courage - TED](https://youtu.be/NDQ1Mi5I4rg)",
            "🎥 [How to Cope with Anxiety](https://youtu.be/hnpQrMqDoqE)",
            "🎥 [Mindfulness for Fear](https://youtu.be/Rl5L7a6Wx_U)",
            "🎥 [Train Your Brain to Overcome Fear](https://youtu.be/1Ojf9l5b2fA)",
            "🎥 [What Fear Can Teach Us](https://youtu.be/vRZ9riM4i0I)"
        ],
        "music": [
            "🎧 [Peaceful Piano](https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO)",
            "🎧 [Calm Meditation](https://open.spotify.com/playlist/37i9dQZF1DX3PIPIT6lEg5)",
            "🎧 [Nature Sounds](https://open.spotify.com/playlist/37i9dQZF1DWXe9gFZP0gtP)",
            "🎧 [Relax & Unwind](https://open.spotify.com/playlist/37i9dQZF1DWUvHZA1zLcjW)",
            "🎧 [Deep Focus](https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ)"
        ],
        "tv": [
            "🎙️ *The Anxious Achiever*",
            "🎙️ *Not Another Anxiety Show*",
            "🎙️ *Anxiety Coaches Podcast*",
            "🎙️ *The Calm Collective*",
            "🎙️ *The Anxiety Slayer*"
        ]
    },
    "neutral": {
        "books": [
            "📖 *Atomic Habits* by James Clear",
            "📖 *The Power of Now* by Eckhart Tolle",
            "📖 *Deep Work* by Cal Newport",
            "📖 *Digital Minimalism* by Cal Newport",
            "📖 *Essentialism* by Greg McKeown",
        ],
        "videos": [
            "🎥 [How to Stay Calm Under Pressure - TED](https://youtu.be/2fjcJp_Nwvk)",
            "🎥 [Do Nothing for 2 Minutes](https://youtu.be/G84xUbeH6rQ)",
            "🎥 [The Power of Boredom](https://youtu.be/Nu0IF2aAqvQ)",
            "🎥 [The Surprising Benefits of Doing Nothing](https://youtu.be/FA2aKOoQbbU)",
            "🎥 [Motivation for Daily Focus](https://youtu.be/E7YzkOZ2sS0)"
        ],
        "music": [
            "🎧 [Focus Flow](https://open.spotify.com/playlist/37i9dQZF1DX6VdMW310YC7)",
            "🎧 [Lo-Fi Beats](https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn)",
            "🎧 [Instrumental Study](https://open.spotify.com/playlist/37i9dQZF1DX8Uebhn9wzrS)",
            "🎧 [Chill Study Beats](https://open.spotify.com/playlist/37i9dQZF1DWTwnEm1IYyoj)",
            "🎧 [Coding Mode](https://open.spotify.com/playlist/37i9dQZF1DX8NTLI2TtZa6)"
        ],
        "tv": [
            "🎙️ *The Tim Ferriss Show*",
            "🎙️ *Daily Stoic*",
            "🎙️ *Mindset Mentor*",
            "🎙️ *The Minimalists*",
            "🎙️ *The Daily Boost*"
        ]
    },
    "surprise": {
        "books": [
            "📖 *Big Magic* by Elizabeth Gilbert",
            "📖 *Originals* by Adam Grant",
            "📖 *The Creative Habit* by Twyla Tharp",
            "📖 *Steal Like an Artist* by Austin Kleon",
            "📖 *Show Your Work!* by Austin Kleon",
        ],
        "videos": [
            "🎥 [Where Good Ideas Come From - TED](https://youtu.be/NugRZGDbPFU)",
            "🎥 [The Power of Surprise](https://youtu.be/AvdBTYGiZ6g)",
            "🎥 [Embrace the Shake](https://youtu.be/a8zPBrHt71E)",
            "🎥 [How Frustration Can Spark Creativity](https://youtu.be/6TWJaFD6R2s)",
            "🎥 [Unexpected Lessons from the Animal World](https://youtu.be/Nt8xyji-DiA)"
        ],
        "music": [
            "🎧 [Discovery Weekly](https://open.spotify.com/playlist/37i9dQZEVXcF1nIl9gqL9y)",
            "🎧 [Indie Shuffle](https://open.spotify.com/playlist/37i9dQZF1DWUS3jbm4YExP)",
            "🎧 [Fresh Finds](https://open.spotify.com/playlist/37i9dQZF1DX4TnpT6PRuPl)",
            "🎧 [New Music Friday](https://open.spotify.com/playlist/37i9dQZF1DX4JAvHpjipBk)",
            "🎧 [Curated Randomness](https://open.spotify.com/playlist/37i9dQZF1DWXLeA8Omikj7)"
        ],
        "tv": [
            "🎙️ *99% Invisible*",
            "🎙️ *Stuff You Should Know*",
            "🎙️ *Radiolab*",
            "🎙️ *Curiosity Daily*",
            "🎙️ *Hidden Brain*"
        ]
    }
}

def extract_url(item):
    match = re.search(r'\((https?://[^\s]+)\)', item)
    return match.group(1) if match else "#"

def extract_title(item):
    match = re.search(r"\*([^*]+)\*", item)
    return match.group(1).strip() if match else "Unknown Title"

def extract_youtube_id(item):
    url = extract_url(item)
    if not url:
        return ""
    match = re.search(r"(?:v=|\/embed\/|\.be\/)([\w-]{11})", url)
    return match.group(1) if match else ""


def extract_spotify_id(item):
    url = extract_url(item)
    match = re.search(r'playlist/([\w\d]+)', url)
    return match.group(1) if match else "37i9dQZF1DXcBWIGoYBM5M"

def extract_cover(item):
    match = re.search(r'\|\s*(https?://\S+\.(jpg|jpeg|png))', item)
    return match.group(1) if match else "https://via.placeholder.com/100"


def get_random_recommendations(emotion):
    pool = emotion_recommendations_pool.get(emotion.lower())
    if not pool:
        return None

    book = random.choice(pool["books"])
    video = random.choice(pool["videos"])
    music = random.choice(pool["music"])
    tv = random.choice(pool["tv"])

    return {
        "book": {
            "title": extract_title(book),
            "link": extract_url(book),
            "cover": extract_cover(book)
        },
        "videoId": extract_youtube_id(video),
        "spotify": extract_spotify_id(music),
        "tv": {
            "link": extract_url(tv),
            "thumbnail": extract_cover(tv)
        }
    }

# === Generate Contextual Encouragement Using Gemini ===
def generate_encouragement_gemini(emotion, sentence):
    prompt = (
        f"This person felt {emotion} because: \"{sentence}\"\n"
        "Your tone should be grounded and a slightly funny—never preachy or robotic. Keep your response to 13 sentences."
        "Be self-aware that you are an web-tool, and make slight jabs at yourself"
        "Acknowledge what they went through, and"
        "Name the emotional layer : Gently reflect what they might be feeling, without assuming too much."
        "Offer subtle reframing : Help them view parts of their experience through a softer or more self-compassionate lens."
        "Introduce a quiet insight : Add a non-obvious reflection that gives their day a bit more meaning or context."
        "Suggest a gentle action : If it fits naturally, offer a calming or uplifting next step (like rest, journaling, music, etc.)"
        "Bring in warmth or humor : Say something that adds lightness, relatability, or makes them smile."
        "Leave them with a grounding reminder : End with something they can carry with them, like a kind phrase or calming truth"
        "End the conversation that relates to the the situation they spoke about"
    )

    try:
        response = client.models.generate_content(
            model="models/gemini-1.5-flash",   # or "models/gemini-1.5-pro"
            contents=prompt  # ✅ Just a string
        )
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"⚠️ Gemini API Error: {e}")
        return "There was an issue generating advice. Please try again later."

# === Analysis Logic ===
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Empty input."}), 400

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    emotions = []
    for s in sentences:
        result = emotion_pipeline(s)[0]
        top_emotion = max(result, key=lambda x: x['score'])['label']
        emotions.append(top_emotion)

    dominant_emotion = Counter(emotions).most_common(1)[0][0]
    recommendations = get_random_recommendations(dominant_emotion)
    advice = generate_encouragement_gemini(dominant_emotion, text)

    return jsonify({
        "advice": advice,
        "emotion": dominant_emotion,
        **recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)