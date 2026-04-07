#!/usr/bin/env python3
"""
4h Donchian breakout with 1d trend filter and volume confirmation
Long when price breaks above Donchian upper band AND 1d EMA20 > EMA50 AND volume > 1.5x avg volume
Short when price breaks below Donchian lower band AND 1d EMA20 < EMA50 AND volume > 1.5x avg volume
Exit when price breaks opposite Donchian band (exit on opposite breakout)
Designed for trending markets with volume confirmation, works in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # === Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume average (20) ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False).mean().values
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            vol_confirm = volume[i] > 1.5 * avg_volume[i]
            
            # Long: breakout above upper band + 1d EMA20 > EMA50 + volume confirmation
            if close[i] > highest_high[i] and ema_20_aligned[i] > ema_50_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short: breakout below lower band + 1d EMA20 < EMA50 + volume confirmation
            elif close[i] < lowest_low[i] and ema_20_aligned[i] < ema_50_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals