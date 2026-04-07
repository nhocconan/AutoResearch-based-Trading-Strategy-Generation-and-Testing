#!/usr/bin/env python3
"""
Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
Long when price breaks above 20-day Donchian upper band with expanding volume AND weekly EMA trend up
Short when price breaks below 20-day Donchian lower band with expanding volume AND weekly EMA trend down
Exit when price crosses back to 20-day moving average
Designed for low frequency (7-25 trades/year) to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
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
    
    # === 20-day Donchian Channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # === Volume confirmation (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === Weekly trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < donch_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > donch_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND weekly trend filter
            if close[i] > donch_upper[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                # Breakout above upper channel with rising weekly EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_lower[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                # Breakdown below lower channel with falling weekly EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals