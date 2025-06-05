import React, { useState } from "react";

export default function MoodRecommendationApp() {
  const [input, setInput] = useState("");
  const [advice, setAdvice] = useState(null);
  const [recommendations, setRecommendations] = useState(null);

  const handleSubmit = () => {
    // Mocked output — replace with API call later
    setAdvice("It's okay to feel overwhelmed sometimes. Just take it one step at a time.");
    setRecommendations({
      video: "🎥 [Celebrate the Small Wins](https://youtu.be/f0LU6i8z1hQ)",
      music: "🎧 [Feel-Good Hits](https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC)",
      podcast: "🎙️ The Happiness Lab",
      book: "📖 *The Book of Awesome* by Neil Pasricha"
    });
  };

  return (
    <div className="min-h-screen bg-blue-100 p-6 flex flex-col items-center gap-6">
      {!advice ? (
        <div className="bg-white p-6 rounded-xl shadow w-full max-w-2xl">
          <h2 className="text-2xl font-semibold text-center mb-4">How was your day?</h2>
          <textarea
            className="w-full p-4 rounded-md border border-gray-300 shadow"
            rows={6}
            placeholder="Type or paste your journal here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button
            className="mt-4 bg-blue-500 hover:bg-blue-600 text-white py-2 px-6 rounded shadow w-full"
            onClick={handleSubmit}
          >
            Analyze & Get Recommendations
          </button>
        </div>
      ) : (
        <>
          <div className="bg-blue-400 text-white rounded-xl p-6 w-full max-w-3xl text-center text-xl shadow">
            <strong>Advice:</strong> <br />
            {advice}
          </div>

          <div className="grid grid-cols-2 gap-6 max-w-3xl w-full">
            <div className="bg-blue-500 text-white rounded-xl p-6 shadow text-center">
              <strong>Video</strong>
              <p className="mt-2">{recommendations.video}</p>
            </div>
            <div className="bg-blue-500 text-white rounded-xl p-6 shadow text-center">
              <strong>Music</strong>
              <p className="mt-2">{recommendations.music}</p>
            </div>
            <div className="bg-blue-500 text-white rounded-xl p-6 shadow text-center">
              <strong>Podcast</strong>
              <p className="mt-2">{recommendations.podcast}</p>
            </div>
            <div className="bg-blue-500 text-white rounded-xl p-6 shadow text-center">
              <strong>Book</strong>
              <p className="mt-2">{recommendations.book}</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
