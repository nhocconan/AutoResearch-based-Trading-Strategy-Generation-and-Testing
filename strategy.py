#!/usr/bin/env python3
# 12h_1d_rsi_divergence_volume_v1
# Strategy: 12h RSI divergence with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: RSI divergence identifies exhaustion points in trends. Combined with volume confirmation and daily trend filter, it provides high-probability reversal entries in both bull and bear markets. Designed for low frequency (15-25 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_rsi_divergence_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI calculation (14-period)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # Price peaks and troughs for divergence detection
    # Look for local maxima and minima over 5-period window
    def find_local_extrema(arr, window=5):
        maxima = np.zeros(len(arr), dtype=bool)
        minima = np.zeros(len(arr), dtype=bool)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                maxima[i] = True
            if arr[i] == np.min(arr[i-window:i+window+1]):
                minima[i] = True
        return maxima, minima
    
    price_max, price_min = find_local_extrema(close, window=3)
    rsi_max, rsi_min = find_local_extrema(rsi.values, window=3)
    
    # Bearish divergence: price makes higher high, RSI makes lower high
    bullish_divergence = np.zeros(n, dtype=bool)
    bearish_divergence = np.zeros(n, dtype=bool)
    
    for i in range(10, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if price_min[i]:
            # Look back for previous price low
            for j in range(max(0, i-20), i):
                if price_min[j] and close[i] < close[j] and rsi.values[i] > rsi.values[j]:
                    bullish_divergence[i] = True
                    break
        # Bearish divergence: price makes higher high, RSI makes lower high
        if price_max[i]:
            # Look back for previous price high
            for j in range(max(0, i-20), i):
                if price_max[j] and close[i] > close[j] and rsi.values[i] < rsi.values[j]:
                    bearish_divergence[i] = True
                    break
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi.iloc[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: RSI divergence + volume spike + trend alignment for reversal
        if bullish_divergence[i] and volume_spike[i] and not uptrend and position != 1:
            # Bullish divergence in downtrend -> long reversal
            position = 1
            signals[i] = 0.25
        elif bearish_divergence[i] and volume_spike[i] and not downtrend and position != -1:
            # Bearish divergence in uptrend -> short reversal
            position = -1
            signals[i] = -0.25
        # Exit: opposite divergence or trend resumption
        elif position == 1 and (bearish_divergence[i] or uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bullish_divergence[i] or downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals