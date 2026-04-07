#/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and 1d Trend Filter
Long when price breaks above Donchian upper (20-bar high) with expanding volume AND 1d EMA trend up
Short when price breaks below Donchian lower (20-bar low) with expanding volume AND 1d EMA trend down
Exit when price crosses back to Donchian middle (10-bar median)
Designed for 12h timeframe with controlled trade frequency (target: 50-150 trades over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-bar high/low, 10-bar mid) ===
    # Upper: 20-period high
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower: 20-period low  
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle: 10-period median of high/low (more responsive than average)
    median_high = pd.Series(high).rolling(window=10, min_periods=10).median().values
    median_low = pd.Series(low).rolling(window=10, min_periods=10).median().values
    donchian_mid = (median_high + median_low) / 2
    
    # === ATR (14) for volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any NaN values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND 1d trend filter
            if close[i] > highest_high[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper channel with rising 1d EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower channel with falling 1d EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals