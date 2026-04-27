#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Spike.
Long when: 1) Price breaks above Donchian upper (20), 2) Price > 1d EMA50 (bullish trend), 3) Volume > 2x 20-period average.
Short when: 1) Price breaks below Donchian lower (20), 2) Price < 1d EMA50 (bearish trend), 3) Volume > 2x 20-period average.
Exit when price returns to Donchian middle (mean reversion) or trend reverses.
Designed for 12h timeframe: targets 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i-donchian_window+1:i+1])
        lower[i] = np.min(low[i-donchian_window+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), 1d EMA50 (50), volume MA (20)
    start_idx = max(donchian_window, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_channel = upper[i]
        lower_channel = lower[i]
        middle_channel = middle[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper channel + bullish trend + volume spike
            if price > upper_channel and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower channel + bearish trend + volume spike
            elif price < lower_channel and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (mean reversion) or trend turns bearish
            if price <= middle_channel or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle (mean reversion) or trend turns bullish
            if price >= middle_channel or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0