#!/usr/bin/env python3
"""
1h_RSI_Trend_Zone_v1
RSI(14) + 4h/1d trend filter with session filter (08-20 UTC).
Long: RSI<35 + price above 4h EMA20 + price above 1d EMA50
Short: RSI>65 + price below 4h EMA20 + price below 1d EMA50
Exit: RSI crosses back to neutral zone (45-55)
Target: 60-150 total trades over 4 years (15-37/year) with ~0.20 position size.
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
    
    # === RSI(14) ===
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h EMA20 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # pre-compute before loop
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI<35 + price above 4h EMA20 + price above 1d EMA50
            if (rsi[i] < 35 and 
                close[i] > ema_20_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: RSI>65 + price below 4h EMA20 + price below 1d EMA50
            elif (rsi[i] > 65 and 
                  close[i] < ema_20_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI > 45 (return to neutral)
            if rsi[i] > 45:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 55 (return to neutral)
            if rsi[i] < 55:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Trend_Zone_v1"
timeframe = "1h"
leverage = 1.0