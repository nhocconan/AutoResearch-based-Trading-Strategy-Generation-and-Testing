#!/usr/bin/env python3
"""
12h Donchian Breakout + 1d Trend + Volume Confirmation
Long when price breaks above Donchian(20) high and close > 1d EMA50 with volume > 1.5x average
Short when price breaks below Donchian(20) low and close < 1d EMA50 with volume > 1.5x average
Exit on opposite Donchian breakout or trend reversal
Designed to capture sustained moves with trend alignment and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === Volume Average (20) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reversal (close < EMA50)
            if low[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reversal (close > EMA50)
            if high[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout conditions with trend and volume confirmation
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            if high[i] > highest_high[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif low[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals