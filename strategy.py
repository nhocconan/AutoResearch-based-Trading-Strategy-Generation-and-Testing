#!/usr/bin/env python3
"""
6h_KAMA_Trend_Filter_Volume_Spike
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 6h data to capture trend direction, filtered by 12h EMA trend and volume spikes (volume > 2x 20-period average). Enter long when price crosses above KAMA with volume confirmation and 12h EMA up; short when price crosses below KAMA with volume confirmation and 12h EMA down. Exit when price crosses back across KAMA. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Volume spikes confirm institutional participation. Designed for 6h timeframe to target 50-150 total trades over 4 years.
"""

name = "6h_KAMA_Trend_Filter_Volume_Spike"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 6h close
    # Parameters: ER length=10, Fast EMA=2, Slow EMA=30
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_len elements
    change_padded = np.concatenate([np.full(er_len, np.nan), change])
    volatility_padded = np.concatenate([np.full(er_len, np.nan), volatility])
    
    # Calculate volatility as sum of absolute changes over er_len period
    volatility_sum = np.convolve(np.abs(np.diff(close)), np.ones(er_len), 'same')
    # Fix edges
    volatility_sum[:er_len-1] = np.nan
    volatility_sum[-er_len+1:] = np.nan
    
    er = np.where(volatility_sum > 0, change_padded / volatility_sum, 0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above KAMA + volume spike + 12h EMA50 up
            if close[i-1] <= kama[i-1] and close[i] > kama[i] and vol_spike and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + volume spike + 12h EMA50 down
            elif close[i-1] >= kama[i-1] and close[i] < kama[i] and vol_spike and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals