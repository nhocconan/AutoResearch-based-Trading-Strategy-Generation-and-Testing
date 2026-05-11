#!/usr/bin/env python3
name = "6h_1w_1d_Momentum_Confluence"
timeframe = "6h"
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
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema50_w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_trend = weekly_close > ema50_w
    
    # Daily RSI(14) for momentum
    daily_close = df_1d['close'].values
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 60-period volume average for confirmation
    vol_ma60 = np.zeros(n)
    for i in range(n):
        if i < 60:
            vol_ma60[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma60[i] = np.mean(volume[i-59:i+1])
    
    # Align weekly trend and daily RSI to 6h
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 60)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + RSI > 55 + volume confirmation
            if (weekly_trend_aligned[i] and 
                rsi_aligned[i] > 55 and 
                volume[i] > 1.3 * vol_ma60[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI < 45 + volume confirmation
            elif (not weekly_trend_aligned[i] and 
                  rsi_aligned[i] < 45 and 
                  volume[i] > 1.3 * vol_ma60[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes or RSI < 40
            if (not weekly_trend_aligned[i] or rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend changes or RSI > 60
            if (weekly_trend_aligned[i] or rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals