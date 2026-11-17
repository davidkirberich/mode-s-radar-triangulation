# mode-s-radar-triangulation

Passive Vermessung von Mode-S-Bodenradaren anhand von `DF11`-All-Call-Antworten (`V1` bis `V5`) sowie Ortung von Flugzeugen ohne ADS-B-Positionsmeldungen per Extended Kalman Filter und RTS-Smoother (`V6`) mit einem einzelnen `1090-MHz`-Empfänger.

Zum vollständige Artikel: https://www.david-kirberich.de/lab/mode-s-radar-triangulation/

## Repository-Struktur

- `src/`: Aufzeichnungs- und Archivierungs-Skripte (`beast_record.py`, `beast_capture.py`, `beast_archive.py`), Kalibrierungs- und Tracking-Skripte (`V1` bis `V6`) sowie gemeinsame Hilfsmodule (`beast_inspect.py`, `experiment_runs.py`)
- `tests/`: Unit-Tests für `V3` bis `V6`
- `results/`: Referenzläufe und Kartenausgaben (`V1` bis `V6`)

## Installation und Tests

Vorausgesetzt werden Linux/Unix (`V6` nutzt `fcntl` für Lockfiles), Python 3.11+ sowie `zstd` für die Segment-Archivierung:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest
```

Einzige externe Laufzeitabhängigkeit für die Auswertung ist `numpy`; `pytest` wird für die Testsuite benötigt.

## Beast-Aufzeichnung und Archivierung

Die Datenerfassung besteht aus [`beast_record.py`](src/beast_record.py) (samt [`beast_capture.py`](src/beast_capture.py)) und [`beast_archive.py`](src/beast_archive.py) (samt [`beast_inspect.py`](src/beast_inspect.py)).

- **Aufzeichnung (`beast_record.py`):** Schreibt den Beast-TCP-Bytestrom unverändert in stündliche `.raw`-Segmente. Eine begleitende `.jsonl`-Datei protokolliert Empfangszeiten, Byte-Offsets, Segment-Metadaten sowie sekündliche `aircraft.json`-Snapshots. Abgeschlossene Stunden-Segmente werden über eine `.closed`-Markerdatei gekennzeichnet.
- **Archivierung (`beast_archive.py`):** Komprimiert geschlossene `.raw`-Segmente mit `zstd` (`.raw.zst`), prüft die Integrität des Archivs und erzeugt pro Segment eine abgeleitete SQLite-Datenbank (`capture-*.sqlite`). Die Datenbank enthält nur die extrahierten Ereignisse und Indizes; der vollständige Bytestrom verbleibt in `.raw` bzw. `.raw.zst`.

### SQLite-Schema

Das Datenbankschema wird in [`beast_archive.py`](src/beast_archive.py) angelegt:

| Tabelle | Inhalt |
| --- | --- |
| `run` | Startmetadaten, Segmentstatus und vollständige Frame-Grenzen |
| `df11_events` | `DF11`-Ereignisse mit Byte-Offset, Zeitstempeln, Beast-Ticks, ICAO-Adresse, II/SI-Code und zugeordnetem Positions-Snapshot |
| `clock_anchors` | Stützpunkte zwischen monotoner Systemzeit und Beast-Ticks |
| `clock_fit` | Geschätzte Tickrate, Residuum und erkannte Taktsprünge |

`recv_ns` und `monotonic_ns` speichern lokale Empfangszeitstempel in Nanosekunden, `mlat_ticks` stammt aus dem Beast-Header und `raw_offset` verweist auf die Byte-Position im unkomprimierten `.raw`-Strom.

### Aufruf der Aufzeichnung

Der Recorder erwartet einen laufenden Dienst (`adsb-receiver-v2`) mit Beast-TCP-Ausgabe auf `127.0.0.1:30005`, `aircraft.json` unter `/dev/shm/adsb-live-v2/aircraft.json` sowie mindestens 5 GiB freien Speicherplatz im Zielverzeichnis. Die Empfängerkoordinaten in `beast_record.py` sind für das ursprüngliche Setup hinterlegt und müssen für andere Standorte angepasst werden.

```sh
python src/beast_record.py --directory /pfad/zu/capture --end-utc UNIX_SECONDS
python src/beast_archive.py --directory /pfad/zu/capture
```

`--end-utc` legt das Aufnahmeende als Unix-Zeitstempel in Sekunden fest. Wird derselbe Parameter an `beast_archive.py` übergeben, deaktiviert das Skript nach Erreichen des Endzeitpunkts den systemd-Timer `adsb-beast-archive.timer`.

## Radarkalibrierung und Tracking (V1–V6)

Alle Auswertungsskripte akzeptieren `--archive`, `--output`, `--rx-lat` und `--rx-lon` sowie alternativ die Umgebungsvariablen `ADSB_ARCHIVE`, `ADSB_OUTPUT`, `ADSB_RX_LAT` und `ADSB_RX_LON` (CLI-Flags haben Vorrang; Standardwerte sind `/var/lib/adsb/beast-experiment`, `/tmp/radar-triangulation/vN` und `51.145, 6.355`). `V1` akzeptiert die Koordinaten für eine einheitliche Schnittstelle, wertet sie intern aber nicht aus.

```sh
python src/radar_triangulation_v5.py \
  --archive /pfad/zu/capture \
  --output /pfad/zu/output/v5 \
  --rx-lat 51.145 --rx-lon 6.355

python src/mode_s_tracker_v6.py \
  --archive /pfad/zu/capture \
  --output /pfad/zu/output/v6 \
  --radars results/v5/radar-triangulation-v5-20261116.csv
```

Jeder Lauf erzeugt unter `--output` ein Zeitstempel-Verzeichnis `runs/<timestamp>-...` samt Quellcode-Snapshot. Ohne `--radars` sucht `V6` nach der neuesten `*-verified.csv` im Elternverzeichnis von `--output` und legt dort einen Segment-Cache (`cache/mode-s-raw/`) ab. Für Empfängerstandorte außerhalb Mitteleuropas müssen Suchgebiet (`LAT_MIN`/`LAT_MAX`, `LON_MIN`/`LON_MAX`) und Kartenbereich entsprechend angepasst werden.

## Referenz-Ergebnisse (`results/`)

Der Ordner `results/` enthält die ausgewerteten Referenzläufe. Zwischenexperimente und technische Vortests wurden nicht übernommen. Die Schemaübersicht in der V6-Zusammenfassung wurde für den Export auf die vier hier verwendeten Tabellen reduziert.

| Version | Quelldatei (`src/`) | Ursprünglicher Lauf | Enthaltene Dateien | Anmerkung |
| --- | --- | --- | --- | --- |
| `v1` | `radar_triangulation_v1.py` | `v1/runs/undated` | `radar-triangulation-v1.csv`, `.svg` | Erste DF11-Triangulation (15 Bodenradare) |
| `v2` | `radar_triangulation_v2.py` | `v2/runs/20261116` | `radar-triangulation-v2-20261116.csv`, `.svg` | Erweiterung um Burst-Bündelung und Mehrfach-Peilung |
| `v3` | `radar_triangulation_v3.py` | `v3/runs/20261116-original` | `radar-triangulation-v3-20261116.csv`, `.svg` | Zwischenexperiment `20261116-normalisiert` ausgeschlossen |
| `v4` | `radar_triangulation_v4.py` | – | – | Algorithmus-Basis für `V5` und `V6`, kein eigener Referenzlauf |
| `v5` | `radar_triangulation_v5.py` | `v5/runs/20261116-1442` | `radar-triangulation-v5-20261116.csv`, `.svg` | Finaler Kalibrierungslauf (10 verifizierte Bodenradare) |
| `v6` | `mode_s_tracker_v6.py` | `v6/runs/20261116-1536-full-archive` | `v6-mode-s-tracks.csv`, `v6-blind-validation.csv`, `v6-run-summary.json`, `v6-tracks-map.svg` | Vortest `20261116-two-segments` ausgeschlossen |

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).
