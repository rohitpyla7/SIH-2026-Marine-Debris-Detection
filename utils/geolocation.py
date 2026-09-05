"""
Navigation and geographic coordinate utilities.
Formats positional telemetry and handles missing coordinate data cleanly.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any


def build_geolocation(
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    depth_m: Optional[float] = None,
    altitude_m: Optional[float] = None,
    heading_deg: Optional[float] = None,
    timestamp: Optional[str] = None,
    mission_id: Optional[str] = None,
    is_simulated: bool = True,
) -> dict:
    """Build standardized location record from telemetry or simulation parameters."""
    has_coords = lat is not None and lon is not None

    if not has_coords:
        return {
            "available": False,
            "status": "UNAVAILABLE",
            "label": "Location telemetry unavailable",
            "lat": None,
            "lon": None,
            "depth_m": depth_m,
            "altitude_m": altitude_m,
            "heading_deg": heading_deg,
            "timestamp": timestamp,
            "mission_id": mission_id,
        }

    label = "Simulated Coordinates" if is_simulated else "GPS Fix"
    return {
        "available": True,
        "status": "SIMULATED" if is_simulated else "REAL",
        "label": label,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6),
        "depth_m": depth_m,
        "altitude_m": altitude_m,
        "heading_deg": heading_deg,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "mission_id": mission_id,
    }


def format_coords_display(geo: dict) -> str:
    """Format coordinate string for dashboard and report presentation."""
    if not geo.get("available"):
        return "Location Unavailable"
    prefix = "[Simulated]" if geo.get("status") == "SIMULATED" else "[GPS]"
    depth_str = f" | {geo['depth_m']}m depth" if geo.get("depth_m") is not None else ""
    return f"{prefix} {geo['lat']:.5f}°, {geo['lon']:.5f}°{depth_str}"
