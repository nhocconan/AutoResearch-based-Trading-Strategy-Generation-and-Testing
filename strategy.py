#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Donchian Breakout with 4h Trend and Volume Confirmation
# Hypothesis: Buy breakouts above 4h Donchian high when 1d trend is up and volume confirms.
# Sell breakdowns below 4h Donchian low when 1d trend is down and volume confirms.
# Uses 4h for signal direction, 1d for trend filter, 1h for entry timing.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_donchian_breakout_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 1h volume > 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if low[i] <= donchian_low_1h[i] or close[i] < ema_50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if high[i] >= donchian_high_1h[i] or close[i] > ema_50_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Breakout in direction of 1d trend with volume confirmation
            if vol_ok:
                if close[i] > ema_50_1h[i]:  # Uptrend
                    if high[i] >= donchian_high_1h[i] and close[i] > donchian_high_1h[i]:
                        position = 1
                        signals[i] = 0.20
                else:  # Downtrend
                    if low[i] <= donchian_low_1h[i] and close[i] < donchian_low_1h[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals