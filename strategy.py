#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high in uptrend (EMA50 > EMA200)
Short when price breaks below Donchian(20) low in downtrend (EMA50 < EMA200)
Exit on opposite breakout or trend reversal
Volume filter: current volume > 1.5x 20-period average
Designed for trend capture with low trade frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # === Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if low[i] < lowest_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if high[i] > highest_high[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Uptrend: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i] and vol_ok:
                # Long breakout
                if high[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.30
            # Downtrend: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i] and vol_ok:
                # Short breakdown
                if low[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals