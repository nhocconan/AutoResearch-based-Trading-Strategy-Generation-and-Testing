#!/usr/bin/env python3
name = "12h_1d_1w_KAMA_Trend_RSI_Volume"
timeframe = "12h"
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
    
    # Get daily and weekly data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily KAMA for trend
    daily_close = df_1d['close'].values
    change = np.abs(np.diff(daily_close, prepend=daily_close[0]))
    volatility = np.sum(np.abs(np.diff(daily_close, prepend=daily_close[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    kama = np.zeros_like(daily_close)
    kama[0] = daily_close[0]
    for i in range(1, len(daily_close)):
        kama[i] = kama[i-1] + sc[i] * (daily_close[i] - kama[i-1])
    kama_trend = daily_close > kama
    
    # Weekly RSI(14) for momentum
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 20-period volume average for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align daily KAMA trend and weekly RSI to 12h
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_trend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily KAMA uptrend + RSI > 55 + volume confirmation
            if (kama_trend_aligned[i] and 
                rsi_aligned[i] > 55 and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily KAMA downtrend + RSI < 45 + volume confirmation
            elif (not kama_trend_aligned[i] and 
                  rsi_aligned[i] < 45 and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: daily KAMA trend changes or RSI < 40
            if (not kama_trend_aligned[i] or rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: daily KAMA trend changes or RSI > 60
            if (kama_trend_aligned[i] or rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals