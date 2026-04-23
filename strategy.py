#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with 1w EMA50 filter and volume confirmation.
Long when KAMA > EMA50(1w) AND 1d volume > 1.5x 20-day average volume.
Short when KAMA < EMA50(1w) AND 1d volume > 1.5x 20-day average volume.
Exit when KAMA crosses below/above EMA50(1w) or volume drops below average.
Uses 1w HTF for trend filter, 1d for KAMA and volume to reduce whipsaws.
Target: 20-60 total trades over 4 years (5-15/year) for 1d timeframe.
KAMA adapts to market noise, EMA50 provides smooth trend, volume confirms conviction.
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    lookback_er = 10
    er = np.full(n, np.nan)
    for i in range(lookback_er, n):
        change = abs(close[i] - close[i - lookback_er])
        volatility = 0
        for j in range(1, lookback_er + 1):
            volatility += abs(close[i - j + 1] - close[i - j])
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 1.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[lookback_er] = close[lookback_er]  # seed
    for i in range(lookback_er + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_er + 1, 50, 20)  # KAMA, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 1d volume > 1.5x 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: KAMA > EMA50(1w) AND volume filter
            if kama_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA < EMA50(1w) AND volume filter
            elif kama_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: KAMA crosses below EMA50 OR volume drops below average
                if kama_val < ema_val or volume[i] < vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: KAMA crosses above EMA50 OR volume drops below average
                if kama_val > ema_val or volume[i] < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_KAMA_EMA50_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0