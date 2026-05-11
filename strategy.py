#!/usr/bin/env python3
name = "1d_KAMA_T13_EMA48_Slope_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Fix volatility calculation for array
    volatility_full = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility_full[i] = np.sum(np.abs(np.diff(close[max(0, i-er_length+1):i+1], n=1)))
    volatility_full[0:er_length] = np.nan
    er = np.where(volatility_full != 0, change / volatility_full, 0)
    er[0:er_length] = np.nan
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # EMA48 for trend filter
    ema48 = pd.Series(close).ewm(span=48, adjust=False, min_periods=48).mean().values
    
    # KAMA slope (5-period change)
    kama_slope = np.diff(kama, n=5)
    kama_slope = np.concatenate([np.full(5, np.nan), kama_slope])
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter (1w EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    trend_up = close > ema_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_length + 5, 48, 20) + 10  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema48[i]) or np.isnan(kama_slope[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA above EMA48 AND positive slope AND volume filter AND weekly uptrend
            if kama[i] > ema48[i] and kama_slope[i] > 0 and volume[i] > 1.5 * vol_ma20[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA below EMA48 AND negative slope AND volume filter AND weekly downtrend
            elif kama[i] < ema48[i] and kama_slope[i] < 0 and volume[i] > 1.5 * vol_ma20[i] and not trend_up[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA crosses below EMA48 or slope turns negative
            if kama[i] < ema48[i] or kama_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA crosses above EMA48 or slope turns positive
            if kama[i] > ema48[i] or kama_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals