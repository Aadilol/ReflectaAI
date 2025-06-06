from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from google import genai
from collections import Counter
from collections import defaultdict
import random
import re

# === Gemini Client Setup ===
api_key = "AIzaSyCfQF-IORV8C6NH_k0FcYUyuicsTXH5eUg"
client = genai.Client(api_key=api_key)

# === Emotion Detection Model ===
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
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
            "[Channel Orange](https://open.spotify.com/album/392p3shh2jkxUxY2VHvlH8?si=WGJQIF0YQwCdJlfeiIT_RQ)",
            "[Isolation](https://open.spotify.com/album/4EPQtdq6vvwxuYeQTrwDVY)",
            "[Bando Stone & the New World](https://open.spotify.com/album/4yUqNSK6jMi7Y6eWl03U5r)",
            "[Sunburn](https://open.spotify.com/album/2T7LuxZRr6SQMgABLtoYTH?si=MPNEhosSSfumMoAj17Y5xw)",
            "[Luv 4 Rent](https://open.spotify.com/album/6dtDTbVBQ9QwsNaqEnjsOT?si=UI-fZevfQ-WZQc5XKB7Q_g)"
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
            "*The Midnight Library* (https://www.goodreads.com/book/show/52578297-the-midnight-library?from_search=true&from_srp=true&qid=vKC9myftMZ&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1602190253i/52578297.jpg",
            "*It's Kind of a Funny Story* (https://www.goodreads.com/book/show/248704.It_s_Kind_of_a_Funny_Story?from_search=true&from_srp=true&qid=v92iPzyVam&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1420629730i/248704.jpg",
            "*Tuesdays with Morrie & the Five People You Meet in Heaven* (https://www.goodreads.com/book/show/21064231-tuesdays-with-morrie-the-five-people-you-meet-in-heaven?from_search=true&from_srp=true&qid=666nEEDT3s&rank=2) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1394345722i/21064231.jpg",
            "*Tiny Beautiful Things* (https://www.goodreads.com/book/show/63193458-tiny-beautiful-things?from_search=true&from_srp=true&qid=IU28nDHlUU&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1667322306i/63193458.jpg",
            "*When Breath Becomes Air* (https://www.goodreads.com/book/show/25899336-when-breath-becomes-air?from_search=true&from_srp=true&qid=zEyUPYo1kK&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1492677644i/25899336.jpg",
        ],
        "videos": [
            "[You Will Never Do Anything Remarkable ](https://www.youtube.com/watch?v=vmIUvp0e1bw&t=4s)",
            "[Loneliness](https://www.youtube.com/watch?v=n3Xv_g3g-mA)",
            "[Tired of Doomscrolling? ](https://www.youtube.com/watch?v=c1nYtX-NUsc)",
            "[Look who’s doing better than you](https://www.youtube.com/watch?v=8yxY5c0tAtc)",
            "[Bo Burnham Tried To Warn Us](https://www.youtube.com/watch?v=I89Lz7CdLuM)"
        ],
        "music": [
            "[Jaago](https://open.spotify.com/album/0iH6pbUrb45P62B82PHDLb?si=okHHFyZ1SFWW38H1AusOEQ)",
            "[An Authors Demise]https://open.spotify.com/album/2CzdjzbdV3i8U8PUjIrspR?si=gMbqMib2TKqjN8srwrJA-A)",
            "[Lahai](https://open.spotify.com/album/0oKro6GftR6X0sk7fVH7T8?si=AqZku_FcTQmPeQuCFaTAUw)",
            "[Blonde](https://open.spotify.com/album/3mH6qwIy9crq0I9YQbOuDf?si=s7cHRsJ6SgyZDINAuOpgJw)",
            "[Grace](https://open.spotify.com/album/7yQtjAjhtNi76KRu05XWFS?si=Xc8LX1uoQ3q4cwna38NUnA)"
        ],
        "tv": [
            "*Frieren: Beyond Journey's End* (https://www.imdb.com/title/tt22248376/?ref_=chttvtp_t_51) | https://m.media-amazon.com/images/M/MV5BZTI4ZGMxN2UtODlkYS00MTBjLWE1YzctYzc3NDViMGI0ZmJmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Fleabag* (https://www.imdb.com/title/tt5687612/?ref_=chttvtp_t_97) | m.media-amazon.com/images/M/MV5BMjA4MzU5NzQxNV5BMl5BanBnXkFtZTgwOTg3MDA5NzM@._V1_.jpg",
            "*Normal People* (https://www.imdb.com/title/tt9059760/?ref_=tt_mlt_t_3) | https://m.media-amazon.com/images/M/MV5BYWUzNjQ2YmYtNWI3Yi00NzNmLWJjYWYtMDFiM2RjYjNjZWZmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Andor* (https://www.imdb.com/title/tt9253284/?ref_=chttvtp_t_2) | https://m.media-amazon.com/images/M/MV5BNGI2MTJjMjUtMTJhOC00YTY2LTg1NjUtMTdmMjg4YTk2YjM5XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*The Bear* (https://www.imdb.com/title/tt14452776/?ref_=chttvtp_t_22) | https://m.media-amazon.com/images/M/MV5BYWZhNDZiMzAtZmZlYS00MWFmLWE2MWEtNDAxZTZiN2U4Y2U2XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
        ]
    },
    "anger": {
        "books": [
            "*So You’ve Been Publicly Shamed* (https://www.goodreads.com/book/show/22571552-so-you-ve-been-publicly-shamed?from_search=true&from_srp=true&qid=FsoYu9LRWc&rank=1 | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1413749614i/22571552.jpg",
            "*The Dance of Anger* (https://www.goodreads.com/book/show/31312.The_Dance_of_Anger?from_search=true&from_srp=true&qid=9mM7g5XatZ&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1388261597i/31312.jpg",
            "*Radical Acceptance* (https://www.goodreads.com/book/show/213181082-radical-acceptance?from_search=true&from_srp=true&qid=afCQ2FdK81&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1715372161i/213181082.jpg",
            "*Fight Club* (https://www.goodreads.com/book/show/36236124-fight-club?from_search=true&from_srp=true&qid=S6gx4lPtmA&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1558216416i/36236124.jpg",
            "*The Things You Can See Only When You Slow Down* (https://www.goodreads.com/book/show/30780006-the-things-you-can-see-only-when-you-slow-down?from_search=true&from_srp=true&qid=ARAvNRAIOm&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1579340833i/30780006.jpg",
        ],
        "videos": [
            "[Anger Is Your Ally](https://www.youtube.com/watch?v=sbVBsrNnBy8)",
            "[Mark Normand | Out To Lunch](https://www.youtube.com/watch?v=v69-181zzk8&t=614s)",
            "[Dave Chappelle | Everything I Say Upsets Somebody](https://www.youtube.com/watch?v=pvc_XDDrwgc&t=1530s)",
            "[Indiana Jones and the Objective Existence of God](https://www.youtube.com/watch?v=LJkMWj7QGcA)",
            "[everybody is a total mess](https://www.youtube.com/watch?v=0y6wiBzPMSI&t=273s)"
        ],
        "music": [
            "[I Lay Down My Life for You](https://open.spotify.com/album/1ezs1QD5SYQ6LtxpC9y5I2?si=8WEIyxnqQ6yxc5NUDeAECA)",
            "[Mr. Morale & the Big Steppers](https://open.spotify.com/album/79ONNoS4M9tfIA1mYLBYVX?si=FnVZkyutQbO-DQ-OotiO3w)",
            "[King of the Mischievous South Vol. 2](https://open.spotify.com/album/6LoDd1G8en4TcqdSg7yqrV?si=nVHeaThARFG13Ke7WlDJVQ)",
            "[Heroes & Villains](https://open.spotify.com/album/4gR3h0hcpE1iJH0v5bVv78?si=0nOUrFSASGyhGsiF1OC3xg)",
            "[Around the Fur](https://open.spotify.com/album/7o4UsmV37Sg5It2Eb7vHzu?si=kYKqQUM_TnqTBPISZ7HK8w)"
        ],
        "tv": [
            "*Arcane* (https://www.imdb.com/title/tt11126994/?ref_=chttvtp_t_25) | https://m.media-amazon.com/images/M/MV5BOWJhYjdjNWEtMWFmNC00ZjNkLThlZGEtN2NkM2U3NTVmMjZkXkEyXkFqcGc@._V1_.jpg",
            "*The Studio* (https://www.imdb.com/title/tt23649128/?ref_=chttvm_t_27) | https://m.media-amazon.com/images/M/MV5BMDQxMWI5OTMtNGRkMC00NTVlLWI5ZjAtZmFiMjMwM2M0N2E0XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Peaky Blinders* (https://www.imdb.com/title/tt2442560/?ref_=chttvm_t_83) | https://m.media-amazon.com/images/M/MV5BOGM0NGY3ZmItOGE2ZC00OWIxLTk0N2EtZWY4Yzg3ZDlhNGI3XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Sherlock* (https://www.imdb.com/title/tt1475582/?ref_=tt_mlt_i_11) | https://m.media-amazon.com/images/M/MV5BNTQzNGZjNDEtOTMwYi00MzFjLWE2ZTYtYzYxYzMwMjZkZDc5XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Shōgun* (https://www.imdb.com/title/tt2788316/?ref_=chttvtp_t_151) | https://www.imdb.com/title/tt2788316/?ref_=chttvtp_t_151",
        ]
    },
    "fear": {
        "books": [
            "*The Boy, the Mole, the Fox and the Horse* (https://www.goodreads.com/book/show/43708884-the-boy-the-mole-the-fox-and-the-horse?from_search=true&from_srp=true&qid=BXdWBYM2ok&rank=1 | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1579017235i/43708884.jpg",
            "*Atlas of the Heart* (https://www.goodreads.com/book/show/58330567-atlas-of-the-heart?from_search=true&from_srp=true&qid=l1mL51AUBS&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1630947399i/58330567.jpg",
            "*The Night Circus* (https://www.goodreads.com/book/show/9361589-the-night-circus?from_search=true&from_srp=true&qid=bq0lApcsfg&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1387124618i/9361589.jpg",
            "*The Body Keeps the Score* (https://www.goodreads.com/book/show/18693771-the-body-keeps-the-score?from_search=true&from_srp=true&qid=iugz9XCya8&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1594559067i/18693771.jpg",
            "*Station Eleven* (https://www.goodreads.com/book/show/20170404-station-eleven?from_search=true&from_srp=true&qid=6eK8zwEkEX&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1680459872i/20170404.jpg",
        ],
        "videos": [
            "[8 Million Species of Aliens](https://www.youtube.com/watch?v=DvkUo05xtFM&t=25s)",
            "[True Stories About Chimps](https://www.youtube.com/watch?v=Yb5D9Q4gZH8)",
            "[What Will We Miss?](https://www.youtube.com/watch?v=7uiv6tKtoKg)",
            "[Did The Future Already Happen? - The Paradox of Time](https://www.youtube.com/watch?v=wwSzpaTHyS8)",
            "[Fear of Dark](https://www.youtube.com/watch?v=BcQ-8R2fiZg)"
        ],
        "music": [
            "[Portrait Of A Time](https://open.spotify.com/album/2zQLWuvxPSb572kjmeAEYN?si=UbifXDL9TF6xCYaJu4DhHQ)",
            "[Gemini Rights](https://open.spotify.com/album/3Ks0eeH0GWpY4AU20D5HPD?si=KajazEX1SCejtJoQFWmxsw)",
            "[Let's Start Here](https://open.spotify.com/album/6Per97deaWqrJlKQNX8RGK?si=cxCc_clAR9-oRYOXEPHt5Q)",
            "[Dawn](https://open.spotify.com/album/3CogjJSvRqbIQuNJVR2JcP?si=HD49sQIYRxGPA0mLLDeQYA)",
            "[Why Lawd?](https://open.spotify.com/album/0LlzHi8Erl8zpTumqt88Qe?si=QkpX-D0bRGGf27CAtiXFCQ)"
        ],
        "tv": [
            "*Planet Earth* (https://www.imdb.com/title/tt0795176/?ref_=chttvtp_t_3) | https://m.media-amazon.com/images/M/MV5BY2NjNDUzOTgtMDFmNC00ZGQ4LWE5MDctMzczNGVlOGU1N2MyXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Cosmos: A Spacetime Odyssey* (https://www.imdb.com/title/tt2395695/?ref_=chttvtp_t_10) | https://m.media-amazon.com/images/M/MV5BYTRlMzk0NzctNTI3Ni00N2E2LWJiNGMtMDdlNjk1YWNmMzkyXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*True Detective* (https://www.imdb.com/title/tt2356777/?ref_=chttvtp_t_45) | https://m.media-amazon.com/images/M/MV5BYjgwYzA1NWMtNDYyZi00ZGQyLWI5NTktMDYwZjE2OTIwZWEwXkEyXkFqcGc@._V1_.jpg",
            "*BoJack Horseman* (https://www.imdb.com/title/tt3398228/?ref_=chttvtp_t_64) | https://m.media-amazon.com/images/M/MV5BZmMwMDlkNTEtMmQzZS00ODQ0LWJlZmItOTgwYWMwZGM4MzFiXkEyXkFqcGc@._V1_.jpg",
            "*Life* (https://www.imdb.com/title/tt1533395/?ref_=chttvtp_t_17) | https://m.media-amazon.com/images/M/MV5BZDJjMzJiMTktMWZkZi00YWY0LWJjNGUtY2ZmNTFlOThhZTA4XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
        ]
    },
    "neutral": {
        "books": [
            "*The Midnight Library* (https://www.goodreads.com/book/show/52578297-the-midnight-library?from_search=true&from_srp=true&qid=vKC9myftMZ&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1602190253i/52578297.jpg",
            "*So You’ve Been Publicly Shamed* (https://www.goodreads.com/book/show/22571552-so-you-ve-been-publicly-shamed?from_search=true&from_srp=true&qid=FsoYu9LRWc&rank=1 | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1413749614i/22571552.jpg",
            "*The Boy, the Mole, the Fox and the Horse* (https://www.goodreads.com/book/show/43708884-the-boy-the-mole-the-fox-and-the-horse?from_search=true&from_srp=true&qid=BXdWBYM2ok&rank=1 | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1579017235i/43708884.jpg",
            "*The Midnight Library* (https://www.goodreads.com/book/show/52578297-the-midnight-library?from_search=true&from_srp=true&qid=vKC9myftMZ&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1602190253i/52578297.jpg",
            "*Tuesdays with Morrie & the Five People You Meet in Heaven* (https://www.goodreads.com/book/show/21064231-tuesdays-with-morrie-the-five-people-you-meet-in-heaven?from_search=true&from_srp=true&qid=666nEEDT3s&rank=2) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1394345722i/21064231.jpg",
        ],
        "videos": [
            "[8 Million Species of Aliens](https://www.youtube.com/watch?v=DvkUo05xtFM&t=25s)",
            "[Mark Normand | Out To Lunch](https://www.youtube.com/watch?v=v69-181zzk8&t=614s)",
            "[Did The Future Already Happen? - The Paradox of Time](https://www.youtube.com/watch?v=wwSzpaTHyS8)",
            "[Community having world-class writing for 26 minutes straight](https://www.youtube.com/watch?v=efbSijkAwKI)",
            "[Teletubbies Reunion ](https://www.youtube.com/watch?v=jG41tYE-A-4)"
        ],
        "music": [
            "[Lahai](https://open.spotify.com/album/0oKro6GftR6X0sk7fVH7T8?si=AqZku_FcTQmPeQuCFaTAUw)",
            "[Why Lawd?](https://open.spotify.com/album/0LlzHi8Erl8zpTumqt88Qe?si=QkpX-D0bRGGf27CAtiXFCQ)"
            "[I Lay Down My Life for You](https://open.spotify.com/album/1ezs1QD5SYQ6LtxpC9y5I2?si=8WEIyxnqQ6yxc5NUDeAECA)",
            "[Grace](https://open.spotify.com/album/7yQtjAjhtNi76KRu05XWFS?si=Xc8LX1uoQ3q4cwna38NUnA)"
            "[Let's Start Here](https://open.spotify.com/album/6Per97deaWqrJlKQNX8RGK?si=cxCc_clAR9-oRYOXEPHt5Q)",
        ],
        "tv": [
            "*True Detective* (https://www.imdb.com/title/tt2356777/?ref_=chttvtp_t_45) | https://m.media-amazon.com/images/M/MV5BYjgwYzA1NWMtNDYyZi00ZGQyLWI5NTktMDYwZjE2OTIwZWEwXkEyXkFqcGc@._V1_.jpg",
            "*The Studio* (https://www.imdb.com/title/tt23649128/?ref_=chttvm_t_27) | https://m.media-amazon.com/images/M/MV5BMDQxMWI5OTMtNGRkMC00NTVlLWI5ZjAtZmFiMjMwM2M0N2E0XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Andor* (https://www.imdb.com/title/tt9253284/?ref_=chttvtp_t_2) | https://m.media-amazon.com/images/M/MV5BNGI2MTJjMjUtMTJhOC00YTY2LTg1NjUtMTdmMjg4YTk2YjM5XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Blackadder Goes Forth* (https://www.imdb.com/title/tt0096548/?ref_=chttvtp_t_70) | https://m.media-amazon.com/images/M/MV5BM2ZiODg3ZWQtMzcyMC00MTRhLWI2MjItNjk5OTdlOTRiMGRiXkEyXkFqcGc@._V1_.jpg",
            "*Frieren: Beyond Journey's End* (https://www.imdb.com/title/tt22248376/?ref_=chttvtp_t_51) | https://m.media-amazon.com/images/M/MV5BZTI4ZGMxN2UtODlkYS00MTBjLWE1YzctYzc3NDViMGI0ZmJmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
        ]
    },
    "surprise": {
        "books": [
            "*Gone Girl* (https://www.goodreads.com/book/show/19288043-gone-girl?from_search=true&from_srp=true&qid=EPxXzSlHID&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1554086139i/19288043.jpg",
            "*Klara and the Sun* (https://www.goodreads.com/book/show/54120408-klara-and-the-sun?ac=1&from_search=true&qid=vNVGgGPBjx&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1603206535i/54120408.jpg",
            "*Cloud Atlas* (https://www.goodreads.com/book/show/49628.Cloud_Atlas?from_search=true&from_srp=true&qid=ESJV7Y4r7v&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1563042852i/49628.jpg",
            "*Sapiens* (https://www.goodreads.com/book/show/23692271-sapiens?from_search=true&from_srp=true&qid=dS5SF6pRZY&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1703329310i/23692271.jpg",
            "*Thinking, Fast and Slow* (https://www.goodreads.com/book/show/11468377-thinking-fast-and-slow?from_search=true&from_srp=true&qid=8M4G87gWEL&rank=1) | https://images-na.ssl-images-amazon.com/images/S/compressed.photo.goodreads.com/books/1317793965i/11468377.jpg",
        ],
        "videos": [
            "[There's No Such Thing As Orange](https://www.youtube.com/watch?v=WX0xWJpr0FY)",
            "[The Great Silence](https://www.youtube.com/watch?v=ryg077wBvsM)",
            "[The Beautiful Horror of Deep Space](https://www.youtube.com/watch?v=iqnpZngxYMs&t=75s)",
            "[AI is ruining the internet](https://www.youtube.com/watch?v=UShsgCOzER4)",
            "[Why do we ask questions?](https://www.youtube.com/watch?v=u9hauSrihYQ)"
        ],
        "music": [
            "[Electric Dusk](https://open.spotify.com/album/5u7OrPu6BbadcZNWuH10VT?si=4MP2GElzRvCwIljmZ2upqw)",
            "[IGOR](https://open.spotify.com/album/5zi7WsKlIiUXv09tbGLKsE?si=RxA5VncHSD6kFy-THBpcqA)",
            "[Ego Death](https://open.spotify.com/album/69g3CtOVg98TPOwqmI2K7Q?si=OPzHETfSQGW7pkbagvAisA)",
            "[Purple Rain](https://open.spotify.com/album/2umoqwMrmjBBPeaqgYu6J9?si=GsA173E7QiWkHBDOsLBs7A)",
            "[Honeybloom](https://open.spotify.com/album/6CwBR2GHX84xyA8T95HTM3?si=YY_yJU4pSLiQmzzvYjXHog)"
        ],
        "tv": [
            "*Attack on Titan* (https://www.imdb.com/title/tt2560140/?ref_=chttvtp_t_20) | https://m.media-amazon.com/images/M/MV5BZjliODY5MzQtMmViZC00MTZmLWFhMWMtMjMwM2I3OGY1MTRiXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Better Call Saul* (https://www.imdb.com/title/tt3032476/?ref_=chttvtp_t_26) | m.media-amazon.com/images/M/MV5BNDdjNTEzMjMtYjM3Mi00NzQ3LWFlNWMtZjdmYWU3ZDkzMjk1XkEyXkFqcGc@._V1_.jpg",
            "*Severance* (https://www.imdb.com/title/tt11280740/?ref_=nv_sr_srsg_1_tt_7_nm_0_in_0_q_seve) | https://m.media-amazon.com/images/M/MV5BZDI5YzJhODQtMzQyNy00YWNmLWIxMjUtNDBjNjA5YWRjMzExXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*The Handmaid's Tale* (https://www.imdb.com/title/tt5834204/?ref_=tt_mlt_t_11) | m.media-amazon.com/images/M/MV5BMWIxMzk4NmItZmM1YS00ODUyLWFlNjgtZDQ4MzljZTZmZDQ5XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg",
            "*Succession* (https://www.imdb.com/title/tt7660850/?ref_=chttvtp_t_58) | https://m.media-amazon.com/images/M/MV5BNTEwNTFkZTktMzI1OC00YmRjLWE5NTUtYmZhMmJlNGUxMWU1XkEyXkFqcGc@._V1_.jpg",
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


def extract_spotify_album_id(markdown_link):
    match = re.search(r"open\.spotify\.com/album/([a-zA-Z0-9]+)", markdown_link)
    return match.group(1) if match else None

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
        "spotify": extract_spotify_album_id(music),
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
        "Acknowledge what they went through, and"
        "Name the emotional layer : Gently reflect what they might be feeling, without assuming too much."
        "Offer subtle reframing : Help them view parts of their experience through a softer or more self-compassionate lens."
        "Introduce a quiet insight : Add a non-obvious reflection that gives their day a bit more meaning or context."
        "Suggest a gentle action : If it fits naturally, offer a calming or uplifting next step (like rest, journaling, music, etc.)"
        "Bring in warmth or humor : Say something that adds lightness, relatability, or makes them smile."
        "Be self-aware that you are an web-tool, and make slight jabs at yourself"
        "Leave them with a grounding reminder : End with something they can carry with them, like a kind phrase or calming truth"
        "End the conversation with something that relates to the the situation they spoke about"
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

    # Run entire input through the emotion classifier
    results = emotion_pipeline(text)[0]

    # Build a dictionary of scores per emotion
    emotion_scores = {e['label']: e['score'] for e in results}
    #print("Final Emotion Scores:", emotion_scores)  
    # Get the dominant emotion
    if not emotion_scores:
        return jsonify({"error": "No emotions detected."}), 500
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    # Your existing helper functions
    recommendations = get_random_recommendations(dominant_emotion)
    advice = generate_encouragement_gemini(dominant_emotion, text)

    return jsonify({
        "advice": advice,
        "emotion": dominant_emotion,
        **recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)