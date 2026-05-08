#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(14) trend + 1d volume spike + 1w RSI filter
# KAMA adapts to market noise, providing smooth trend direction.
# Volume spike on daily confirms institutional participation.
# Weekly RSI > 50 for longs, < 50 for shorts ensures alignment with higher timeframe momentum.
# Targets 20-30 trades per year (~80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for momentum-aligned trends.

name = "4h_KAMA14_1dVolume_1wRSI"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA on 4h
    er_window = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, n=er_window))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    
    er = np.zeros_like(close)
    for i in range(er_window, n):
        if volatility[i-er_window:i].sum() > 0:
            er[i] = change[i] / volatility[i-er_window:i].sum()
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[er_window] = close[er_window]
    for i in range(er_window + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Get 1w data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = er_window + 20  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA, volume spike, bullish weekly RSI
            if close[i] > kama[i] and vol_spike_1d_aligned[i] and rsi_1w_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, volume spike, bearish weekly RSI
            elif close[i] < kama[i] and vol_spike_1d_aligned[i] and rsi_1w_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or bearish weekly RSI
            if close[i] < kama[i] or rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or bullish weekly RSI
            if close[i] > kama[i] or rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals