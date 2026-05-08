#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Direction_1wTrend_1dVolSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Get daily data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Daily volume MA(20) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # KAMA direction on 12h timeframe
    # Efficiency Ratio
    change = np.abs(np.diff(close, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(close)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: slope over 5 periods
    kama_slope = np.zeros_like(close)
    kama_slope[5:] = (kama[5:] - kama[:-5]) / 5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(kama_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising + weekly uptrend + daily volume spike
            if (kama_slope[i] > 0 and 
                close[i] > ema_21_1w_aligned[i] and
                vol_ratio_1d_aligned[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + weekly downtrend + daily volume spike
            elif (kama_slope[i] < 0 and 
                  close[i] < ema_21_1w_aligned[i] and
                  vol_ratio_1d_aligned[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA falling or weekly downtrend
            if (kama_slope[i] < 0 or close[i] < ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA rising or weekly uptrend
            if (kama_slope[i] > 0 or close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals