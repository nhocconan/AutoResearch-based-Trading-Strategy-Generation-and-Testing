#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_Filter"
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
    
    # 1w trend: close above/below 1w EMA26
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    trend_up = close > ema_1w_aligned
    
    # Daily KAMA with ER=10
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # ER calculation using rolling window
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 9:
            er[i] = 0
        else:
            change_window = np.abs(np.diff(close_1d[i-9:i+1]))
            volatility_window = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            if volatility_window > 0:
                er[i] = change_window[-1] / volatility_window
            else:
                er[i] = 0
    # Smoothing constants
    sc = (er * (2/(10+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily volume filter: volume > 1.3x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.3 * vol_ma20_1d_aligned
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + weekly uptrend + volume filter
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI < 50 + weekly downtrend + volume filter
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below KAMA or RSI < 40 or weekly trend down
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above KAMA or RSI > 60 or weekly trend up
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60 or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals