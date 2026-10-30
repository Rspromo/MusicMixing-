# Studio Autopilot

Music studio ka automation software. Saari processing isi PC par hoti hai.

## Chalana

Portable folder ke andar `StudioAutopilot.exe` par double-click karein. Installation nahi chahiye.
Poora `StudioAutopilot` folder pen drive se kisi bhi 64-bit Windows PC par copy kar sakte hain.

App Edge ya Chrome ki apni alag window mein khulta hai. Window band karne ke do minute baad app khud band ho jata hai, lekin koi kaam chal raha ho to poora hone tak chalta rehta hai.

## Tabs

| Tab | Kaam |
| --- | --- |
| Auto Mix | Stems se mix, master, Vocal Up, Instrumental aur Ableton stems |
| Download | Link se MP3, WAV, MP4 with BPM aur Key |
| Lyric Video | Lyrics nikalna, edit karna, video render karna |
| Video Editor | Clips aur gaane se edit, MP4 aur Premiere XML |
| Auto Tune | Note-wise correction, raag scales, Ableton .als se tracks |
| Studio Tips | Workflow suggestions |
| Settings | Folders, OpenAI key, Pexels key, encoder |

## Zaroori baatein

- Stems ek hi start point se export karein aur master chain hata dein.
- Premiere mein File > Import se XML chunein, phir project save karein.
- Pehli baar lyrics nikalne par Whisper model download hota hai (medium lagbhag 1.5 GB).
- `StudioData\settings.json` mein API keys save hoti hain. Software kisi ko dene se pehle ye file hata dein.
- Spotify ka audio protected hai, isliye track ka naam lekar same gaana YouTube se milaya jata hai. Result check kar lein.
- Download sirf un gaanon ke liye use karein jinka aapko haq hai ya jo sirf reference ke liye hain.

## Source se chalana

```bash
.venv\Scripts\python.exe app.py
```

## Portable build

```bash
powershell -ExecutionPolicy Bypass -File build.ps1
```
