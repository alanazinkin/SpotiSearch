import os
import json
import google.generativeai as genai

# Explicitly configure genai with the API key
os.environ["GEMINI_API_KEY"] = "AIzaSyA0lk0ovNmex4idlIbHQoQ-EaVM-cnJwAw"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")  # Re-instantiate if necessary

def get_song_descriptions_batch(tracks):
    items_text = "\n".join(
        f"{i + 1}. '{t['track_name']}' by {t['track_artist']}"
        for i, t in enumerate(tracks)
    )

    prompt = f"""
You are helping annotate songs.

For each of the songs below, write EXACTLY ONE short description (2–3 sentences)
including mood, genre, energy, instruments, a theoretical origin playlist, and vocal style (male/female).

Do NOT include or recreate lyrics.

Return STRICT JSON ONLY:

[
  {{"description": "..."}},
  ...
]

Songs:
{items_text}
"""

    response = gemini_model.generate_content(
        prompt,
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except:
        print("FAILED JSON:")
        print(text)
        raise

    return [item.get("description", "").strip() for item in data]


def generate_descriptions_for_indices(indices, df, gemini_batch_size=10):
    all_descs = []
    for start in range(0, len(indices), gemini_batch_size):
        batch_idxs = indices[start:start + gemini_batch_size]
        batch_tracks = []

        for idx in batch_idxs:
            row = df.iloc[idx]
            batch_tracks.append({
                "track_name": row["track_name"],
                "track_artist": row.get("track_artist", "Unknown Artist"),
            })

        descs = get_song_descriptions_batch(batch_tracks)
        print("\n=== Gemini Generated Descriptions (Batch) ===")
        for t, d in zip(batch_tracks, descs):
            print(f" \u2022 {t['track_name']} \u2013 {t['track_artist']}")
            print(f"    -> {d}\n")
        all_descs.extend(descs)
    return all_descs