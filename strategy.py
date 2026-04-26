#!/usr/bin/env python3
"""
1d_KAMA_Trend_Volume_Spice
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a trend filter that whipsaws less in chop. 
On 1d timeframe: long when price > KAMA and volume > 1.5x 20-day average volume; short when price < KAMA and volume > 1.5x average volume.
Uses 1-week EMA50 as higher timeframe trend filter to avoid counter-trend trades. Discrete position sizing (0.25) to limit fee drag.
Target: 20-80 trades over 4 years (5-20/year) on 1d timeframe. Works in both bull (trend following) and bear (avoids false breaks via 1w trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for KAMA and EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 5:
        return np.zeros(n)
    
    # --- 1d KAMA (Kaufman Adaptive Moving Average) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2  # smoothed sc
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- 1d 20-period average volume for confirmation ---
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- 1w EMA50 for trend filter ---
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 10 for KAMA, 20 for avg volume, 50 for 1w EMA)
    start_idx = max(10, 20, 50)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama[i]
        ema_1w_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(kama_val) or np.isnan(ema_1w_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price > KAMA with 1w uptrend and volume confirmation
        long_condition = (close_val > kama_val) and (close_val > ema_1w_val) and volume_confirmed
        # Short logic: price < KAMA with 1w downtrend and volume confirmation
        short_condition = (close_val < kama_val) and (close_val < ema_1w_val) and volume_confirmed
        
        # Exit logic: trend reversal or loss of volume confirmation
        exit_long = (close_val < kama_val) or (close_val < ema_1w_val)
        exit_short = (close_val > kama_val) or (close_val > ema_1w_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_Volume_Spice"
timeframe = "1d"
leverage = 1.0