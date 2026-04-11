#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_trend_v1
Breakout above/below 20-period Donchian channels on 4h with:
- Volume confirmation (1.5x 20-period avg)
- Trend filter using 100-period EMA on 4h
- Position size: 0.25
Target: 25-40 trades/year per symbol (100-160 over 4 years)
Works in bull/bear via trend filter
"""

import numpy as np
import pandas as pd
from typing import Tuple
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def calculate_donchian(high: np.ndarray, low: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate upper and lower Donchian channels."""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def calculate_ema(values: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA with proper warmup."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h indicators
    donchian_window = 20
    dc_upper, dc_lower = calculate_donchian(high, low, donchian_window)
    ema_fast = calculate_ema(close, 50)
    ema_slow = calculate_ema(close, 100)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: EMA50 > EMA100 for long, EMA50 < EMA100 for short
        uptrend = ema_fast[i] > ema_slow[i]
        downtrend = ema_fast[i] < ema_slow[i]
        
        # Breakout conditions
        breakout_up = close[i] > dc_upper[i]
        breakdown_down = close[i] < dc_lower[i]
        
        # Entry conditions
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: opposite Donchian band touch
        long_exit = close[i] < dc_lower[i]  # Price touches lower band
        short_exit = close[i] > dc_upper[i]  # Price touches upper band
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals