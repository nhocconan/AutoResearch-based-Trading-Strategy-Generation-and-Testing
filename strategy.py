#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA50 filter and volume confirmation
# Long when: KAMA crosses above EMA50, volume > 2x 20-period average, and price > 1w EMA50
# Short when: KAMA crosses below EMA50, volume > 2x 20-period average, and price < 1w EMA50
# Exit when KAMA crosses back below/above EMA50 or opposite signal
# Uses KAMA for adaptive trend following, effective in both bull (trend continuation) and bear (trend avoidance) markets.
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_KAMA_EMA50_VolumeSpike_1wEMA50_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for KAMA and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate KAMA (adaptive moving average)
    if len(close_1d) >= 10:
        # Efficiency Ratio (ER)
        change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
        volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of absolute changes
        # Avoid division by zero
        er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
        # Smoothing constants
        fast_sc = 2 / (2 + 1)   # EMA(2)
        slow_sc = 2 / (30 + 1)  # EMA(30)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        # Calculate KAMA
        kama = np.full_like(close_1d, np.nan, dtype=float)
        kama[9] = close_1d[9]  # Start after 10 periods
        for i in range(10, len(close_1d)):
            kama[i] = kama[i-1] + sc[i-10] * (close_1d[i] - kama[i-1])
    else:
        kama = np.full(len(close_1d), np.nan)
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get 1w data ONCE before loop for EMA50 filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA crosses above EMA50, volume filter, and above 1w EMA50
            if (kama_aligned[i] > ema_50_1d_aligned[i] and 
                kama_aligned[i-1] <= ema_50_1d_aligned[i-1] and  # Cross above
                volume_filter[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA crosses below EMA50, volume filter, and below 1w EMA50
            elif (kama_aligned[i] < ema_50_1d_aligned[i] and 
                  kama_aligned[i-1] >= ema_50_1d_aligned[i-1] and  # Cross below
                  volume_filter[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA crosses back below EMA50
            if kama_aligned[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA crosses back above EMA50
            if kama_aligned[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals