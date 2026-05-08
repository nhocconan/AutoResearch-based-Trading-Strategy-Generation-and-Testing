#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_KAMA_Direction_Trend_Filter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close to determine trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio: price change over 10 periods divided by sum of absolute changes
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum |close[t] - close[t-1]|
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility
    # Smoothing constants: fastest = 2/(2+1) = 0.67, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama[9] = close_1d[9]  # Start with first value
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 6h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1d volume moving average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d > 0, vol_ma_1d, 1.0)
    vol_ratio_1d = np.nan_to_num(vol_ratio_1d, nan=1.0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 6h volume confirmation - 6-period average volume (1.5 days for 6h timeframe)
    vol_ma_6h = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    vol_ratio_6h = volume / np.where(vol_ma_6h > 0, vol_ma_6h, 1.0)
    vol_ratio_6h = np.nan_to_num(vol_ratio_6h, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(vol_ratio_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + volume confirmation on both timeframes
            if (close[i] > kama_1d_aligned[i] and
                vol_ratio_1d_aligned[i] > 1.5 and
                vol_ratio_6h[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + volume confirmation on both timeframes
            elif (close[i] < kama_1d_aligned[i] and
                  vol_ratio_1d_aligned[i] > 1.5 and
                  vol_ratio_6h[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals