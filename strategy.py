#!/usr/bin/env python3
# 12h_1d_kama_ema_trend_v1
# Strategy: 12h trend following with KAMA trend detection and EMA confirmation from 1d timeframe
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
#             Combined with 1d EMA trend filter, it captures strong trends while avoiding whipsaws.
#             Designed for low frequency (12-25 trades/year) to minimize fee drag in trending markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_ema_trend_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA calculation (adaptive moving average)
    er_len = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    change = np.insert(change, 0, 0)       # align with original index
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Calculate efficiency ratio properly
    price_change = np.abs(np.diff(close, k=10))
    price_change = np.insert(price_change, 0, 0)
    
    volatility_sum = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility_sum[i] = volatility_sum[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Price channels for entry (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20.iloc[i]) or np.isnan(low_20.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        price_above_ema1d = close[i] > ema_50_1d_aligned[i]
        price_below_ema1d = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        if price_above_kama and price_above_ema1d and close[i] > high_20.iloc[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        elif price_below_kama and price_below_ema1d and close[i] < low_20.iloc[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend deterioration
        elif position == 1 and (close[i] < kama[i] or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama[i] or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals