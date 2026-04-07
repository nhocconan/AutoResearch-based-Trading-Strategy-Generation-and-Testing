#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume Confirmation and 1w Trend Filter
Long when price breaks above 20-day Donchian high with above-average volume AND weekly MA trend up
Short when price breaks below 20-day Donchian low with above-average volume AND weekly MA trend down
Exit when price crosses back to 20-day EMA
Designed for low trade frequency (<150 total over 4 years) to minimize fee drag and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_1w_trend_v1"
timeframe = "1d"
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
    
    # === 20-day EMA for exit ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 20-day Donchian channels ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1-week trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below 20-day EMA
            if close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above 20-day EMA
            if close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need above-average volume (at least 1.1x)
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1w trend filter
            if close[i] > donch_high[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                # Breakout above Donchian high with rising 1w EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                # Breakdown below Donchian low with falling 1w EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals