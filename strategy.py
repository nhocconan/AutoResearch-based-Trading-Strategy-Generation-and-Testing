#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to capture trend direction, confirmed by volume spike and filtered by 12h EMA trend. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing strong trends. Works in bull markets (captures uptrends) and bear markets (avoids false signals during consolidations, captures downtrends). Designed for 4h timeframe to target 20-50 trades/year, avoiding fee drag.
"""

name = "4h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: fast=2, slow=30, lookback=10
    fast = 2
    slow = 30
    lookback = 10
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close, n=lookback))  # |close[t] - close[t-lookback]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over lookback period
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    er = change / volatility  # efficiency ratio
    
    # Calculate smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[lookback] = close[lookback]  # seed
    
    for i in range(lookback + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 1, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above KAMA + volume spike + price above 12h EMA50
            if close[i-1] <= kama[i-1] and close[i] > kama[i] and vol_spike and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + volume spike + price below 12h EMA50
            elif close[i-1] >= kama[i-1] and close[i] < kama[i] and vol_spike and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or price below 12h EMA50
            if close[i] < kama[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or price above 12h EMA50
            if close[i] > kama[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals