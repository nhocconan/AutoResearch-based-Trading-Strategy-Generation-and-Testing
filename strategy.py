#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Filter_1dVolume"
timeframe = "1d"
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
    
    # KAMA: efficiency ratio and smoothing constants
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros(n)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trend_up = close > ema_50_1w_aligned
    
    # Daily volume filter: volume > 1.5x 20-day average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # RSI(14) for overbought/oversold filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + weekly uptrend + volume filter + RSI < 70
            if close[i] > kama[i] and trend_up[i] and volume_filter[i] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + weekly downtrend + volume filter + RSI > 30
            elif close[i] < kama[i] and not trend_up[i] and volume_filter[i] and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below KAMA or weekly trend down
            if close[i] < kama[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above KAMA or weekly trend up
            if close[i] > kama[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals