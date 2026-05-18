from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _http_get_json(url: str, params: Dict[str, Any], timeout_s: int = 15) -> Dict[str, Any]:
    query = urlencode(params, doseq=True)
    full_url = f"{url}?{query}"
    request = Request(
        full_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "veritas-weather-tool/1.0",
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_key(ts: str) -> str:
    return ts[:10]


def _wmo_to_it(code: Optional[int]) -> str:
    mapping = {
        0: "sereno",
        1: "prevalentemente sereno",
        2: "parzialmente nuvoloso",
        3: "coperto",
        45: "nebbia",
        48: "nebbia con brina",
        51: "pioviggine debole",
        53: "pioviggine moderata",
        55: "pioviggine intensa",
        56: "pioviggine gelata debole",
        57: "pioviggine gelata intensa",
        61: "pioggia debole",
        63: "pioggia moderata",
        65: "pioggia forte",
        66: "pioggia gelata debole",
        67: "pioggia gelata intensa",
        71: "neve debole",
        73: "neve moderata",
        75: "neve intensa",
        77: "granuli di neve",
        80: "rovesci deboli",
        81: "rovesci moderati",
        82: "rovesci forti",
        85: "rovesci di neve deboli",
        86: "rovesci di neve forti",
        95: "temporale",
        96: "temporale con grandine lieve",
        99: "temporale con grandine forte",
    }
    if code is None:
        return "non disponibile"
    return mapping.get(code, f"codice meteo {code}")


@dataclass
class VineyardWeatherTool:
    """
    Tool meteo per contesto viticolo.

    Usa API Open-Meteo (geocoding + forecast) e restituisce:
    - previsione giornaliera
    - riepilogo testuale/parafrasi meteo per agenti LLM
    """

    default_country_code: str = "IT"
    default_timezone: str = "Europe/Rome"
    default_source: str = "dwd-icon"
    timeout_seconds: int = 15

    GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URLS: Dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.FORECAST_URLS is None:
            self.FORECAST_URLS = {
                "dwd-icon": "https://api.open-meteo.com/v1/dwd-icon",
                "forecast": "https://api.open-meteo.com/v1/forecast",
                "ecmwf": "https://api.open-meteo.com/v1/ecmwf",
            }
        if self.default_source not in self.FORECAST_URLS:
            raise ValueError(
                f"default_source non supportata: {self.default_source}. "
                f"Valori ammessi: {sorted(self.FORECAST_URLS.keys())}"
            )

    def geocode_location(
        self,
        location: str,
        country_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        cc = (country_code or self.default_country_code).upper()
        payload = _http_get_json(
            self.GEOCODING_URL,
            {
                "name": location,
                "count": 5,
                "language": "it",
                "format": "json",
                "countryCode": cc,
            },
            timeout_s=self.timeout_seconds,
        )
        results = payload.get("results") or []
        if not results:
            raise ValueError(f"Nessuna localita trovata per '{location}' (countryCode={cc}).")

        best = results[0]
        return {
            "name": best.get("name"),
            "admin1": best.get("admin1"),
            "country": best.get("country"),
            "country_code": best.get("country_code"),
            "latitude": best.get("latitude"),
            "longitude": best.get("longitude"),
            "timezone": best.get("timezone"),
        }

    def _fetch_forecast_raw(
        self,
        latitude: float,
        longitude: float,
        days: int,
        timezone: str,
        source: str,
    ) -> Dict[str, Any]:
        if source not in self.FORECAST_URLS:
            raise ValueError(
                f"source non supportata: {source}. "
                f"Valori ammessi: {sorted(self.FORECAST_URLS.keys())}"
            )

        url = self.FORECAST_URLS[source]
        forecast_days = max(1, min(days, 7))
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "forecast_days": forecast_days,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_hours",
                    "wind_speed_10m_max",
                ]
            ),
            "hourly": ",".join(
                [
                    "relative_humidity_2m",
                    "temperature_2m",
                    "precipitation",
                ]
            ),
        }
        return _http_get_json(url, params, timeout_s=self.timeout_seconds)

    def _daily_max_humidity_from_hourly(self, hourly: Dict[str, Any]) -> Dict[str, float]:
        times: List[str] = hourly.get("time") or []
        humidities: List[Any] = hourly.get("relative_humidity_2m") or []
        by_day: Dict[str, float] = {}
        for idx, ts in enumerate(times):
            if idx >= len(humidities):
                break
            h = _to_float(humidities[idx])
            if h is None:
                continue
            day = _date_key(ts)
            current = by_day.get(day)
            if current is None or h > current:
                by_day[day] = h
        return by_day

    def get_vineyard_weather_context(
        self,
        location: str,
        days: int = 3,
        country_code: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        timezone: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Restituisce meteo osservabile per il vigneto.

        Args:
            location: Nome localita (es. "Conegliano").
            days: Numero di giorni di forecast (1..7).
            country_code: Filtro ISO2 per geocoding (default "IT").
            latitude: Latitudine opzionale per bypassare geocoding.
            longitude: Longitudine opzionale per bypassare geocoding.
            timezone: Timezone (default "Europe/Rome" o quella della localita trovata).
            source: Sorgente meteo ("dwd-icon", "forecast", "ecmwf").
        """
        resolved_source = source or self.default_source
        resolved_timezone = timezone or self.default_timezone

        resolved_location: Dict[str, Any]
        if latitude is None or longitude is None:
            resolved_location = self.geocode_location(location, country_code=country_code)
            latitude = _to_float(resolved_location.get("latitude"))
            longitude = _to_float(resolved_location.get("longitude"))
            resolved_timezone = timezone or resolved_location.get("timezone") or resolved_timezone
        else:
            resolved_location = {
                "name": location,
                "admin1": None,
                "country": "N/A",
                "country_code": (country_code or self.default_country_code).upper(),
                "latitude": latitude,
                "longitude": longitude,
                "timezone": resolved_timezone,
            }

        if latitude is None or longitude is None:
            raise ValueError("Coordinate non valide: impossibile recuperare il meteo.")

        raw = self._fetch_forecast_raw(
            latitude=latitude,
            longitude=longitude,
            days=days,
            timezone=resolved_timezone,
            source=resolved_source,
        )

        daily = raw.get("daily") or {}
        hourly = raw.get("hourly") or {}
        humidity_by_day = self._daily_max_humidity_from_hourly(hourly)

        times: List[str] = daily.get("time") or []
        t_max: List[Any] = daily.get("temperature_2m_max") or []
        t_min: List[Any] = daily.get("temperature_2m_min") or []
        precip_sum: List[Any] = daily.get("precipitation_sum") or []
        precip_hours: List[Any] = daily.get("precipitation_hours") or []
        wind_max: List[Any] = daily.get("wind_speed_10m_max") or []
        weather_codes: List[Any] = daily.get("weather_code") or []

        day_rows: List[Dict[str, Any]] = []

        for i, day in enumerate(times):
            row_tmax = _to_float(t_max[i]) if i < len(t_max) else None
            row_tmin = _to_float(t_min[i]) if i < len(t_min) else None
            row_precip = _to_float(precip_sum[i]) if i < len(precip_sum) else 0.0
            row_precip_h = _to_float(precip_hours[i]) if i < len(precip_hours) else 0.0
            row_wind = _to_float(wind_max[i]) if i < len(wind_max) else None
            row_code = int(weather_codes[i]) if i < len(weather_codes) and weather_codes[i] is not None else None
            row_humidity = humidity_by_day.get(day)

            day_rows.append(
                {
                    "date": day,
                    "weather": _wmo_to_it(row_code),
                    "temperature_min_c": row_tmin,
                    "temperature_max_c": row_tmax,
                    "precipitation_mm": row_precip,
                    "precipitation_hours": row_precip_h,
                    "humidity_max_percent": row_humidity,
                    "wind_max_kmh": row_wind,
                }
            )

        short_lines = []
        for day_row in day_rows:
            short_lines.append(
                f"{day_row['date']}: {day_row['weather']}, "
                f"T {day_row['temperature_min_c']}/{day_row['temperature_max_c']} C, "
                f"pioggia {day_row['precipitation_mm']} mm, "
                f"umidita max {day_row['humidity_max_percent']}%."
            )

        summary = (
            f"Forecast {len(day_rows)} giorni per {resolved_location.get('name')} "
            f"({resolved_location.get('admin1')}, {resolved_location.get('country_code')}). "
            + " ".join(short_lines)
        )

        return {
            "tool_name": "vineyard_weather_context_tool",
            "source": f"open-meteo:{resolved_source}",
            "location_resolved": resolved_location,
            "timezone": resolved_timezone,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "forecast_days": day_rows,
            "meteo_forecast": summary,
        }


_weather_tool_instance: Optional[VineyardWeatherTool] = None


def initialize_weather_tool(
    default_country_code: str = "IT",
    default_timezone: str = "Europe/Rome",
    default_source: str = "dwd-icon",
    timeout_seconds: int = 15,
) -> VineyardWeatherTool:
    """
    Inizializza il tool meteo una sola volta.
    """
    global _weather_tool_instance
    _weather_tool_instance = VineyardWeatherTool(
        default_country_code=default_country_code,
        default_timezone=default_timezone,
        default_source=default_source,
        timeout_seconds=timeout_seconds,
    )
    return _weather_tool_instance


def get_vineyard_weather_context(
    location: str,
    days: int = 3,
    country_code: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timezone: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tool function da usare negli agenti MASFactory.

    Esempio:
        get_vineyard_weather_context("Conegliano", days=3)
    """
    if _weather_tool_instance is None:
        raise RuntimeError(
            "Weather tool non inizializzato. "
            "Chiama prima initialize_weather_tool(...)."
        )

    return _weather_tool_instance.get_vineyard_weather_context(
        location=location,
        days=days,
        country_code=country_code,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        source=source,
    )


def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Alias semplice per tool-calling LLM.

    Args:
        location: Localita del vigneto.
        days: Giorni di previsione (1..7).
    """
    return get_vineyard_weather_context(location=location, days=days)
