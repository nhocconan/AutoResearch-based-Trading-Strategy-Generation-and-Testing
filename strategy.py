#!/usr/bin/env python3
"""
Daily Donchian Breakout with Volume Confirmation and Weekly Trend Filter
Long when price breaks above upper Donchian channel (20-day high) with volume expansion AND weekly EMA trend up
Short when price breaks below lower Donchian channel (20-day low) with volume expansion AND weekly EMA trend down
Exit when price crosses back to 10-day EMA
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period high/low) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Exit: 10-day EMA ===
    ema_exit = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Weekly trend filter (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_exit[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 10-day EMA
            if close[i] < ema_exit[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 10-day EMA
            if close[i] > ema_exit[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume expansion (above average)
            if vol_ratio[i] < 1.3:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation AND weekly trend filter
            if close[i] > donch_high[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                # Breakout above upper channel with rising weekly EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donch_low[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                # Breakdown below lower channel with falling weekly EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals