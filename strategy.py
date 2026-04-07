#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w Trend Filter
Long when price breaks above Donchian(20) high and 1w EMA50 > EMA200
Short when price breaks below Donchian(20) low and 1w EMA50 < EMA200
Exit when price breaks opposite Donchian level (long exits on low break, short on high break)
Designed to capture trends with clear entry/exit rules
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend"
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
    
    # === Donchian Channel (20) ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1w EMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if low[i] <= low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if high[i] >= high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Bullish trend: EMA50 > EMA200
            if ema_50_aligned[i] > ema_200_aligned[i]:
                # Long entry: price breaks above Donchian high
                if high[i] >= high_20[i]:
                    position = 1
                    signals[i] = 0.30
            # Bearish trend: EMA50 < EMA200
            elif ema_50_aligned[i] < ema_200_aligned[i]:
                # Short entry: price breaks below Donchian low
                if low[i] <= low_20[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals