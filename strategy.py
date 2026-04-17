#!/usr/bin/env python3
"""
1h_RSI20_MultiTF_Filter_v1
RSI(20) with multi-timeframe trend filter (4h EMA50 & 1d EMA200).
Long: RSI < 30 + price > 4h EMA50 + price > 1d EMA200
Short: RSI > 70 + price < 4h EMA50 + price < 1d EMA200
Exit: RSI crosses back to neutral (40-60 range)
Session filter: 08-20 UTC only
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === RSI(20) ===
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[20] = np.mean(gain[1:21])
    avg_loss[20] = np.mean(loss[1:21])
    
    for i in range(21, n):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d EMA200 for higher timeframe trend ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Session filter: 08-20 UTC only ===
    hours = prices.index.hour  # Pre-compute for efficiency
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI < 30 (oversold) + price above both EMAs (uptrend)
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: RSI > 70 (overbought) + price below both EMAs (downtrend)
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses back to neutral (>= 40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses back to neutral (<= 60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI20_MultiTF_Filter_v1"
timeframe = "1h"
leverage = 1.0