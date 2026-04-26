#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_1dVolatilityFilter
Hypothesis: Camarilla R3/S3 breakouts with 4h EMA50 trend filter and 1d ATR-based volatility regime filter.
Only trade breakouts in direction of 4h trend when volatility is elevated (ATR ratio > 1.0).
Designed for 60-150 total trades over 4 years (15-37/year) with discrete position sizing (0.0, ±0.20).
Uses 4h for trend direction and 1d for volatility filtering, 1h for precise entry timing.
Works in both bull and bear markets by combining breakout momentum with trend and volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1h timeframe (based on previous bar's range)
    # Camarilla R3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # Camarilla S3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    close_prev = np.roll(close, 1)
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev[0] = close[0]  # avoid NaN on first bar
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volatility filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # avoid NaN on first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h ATR(14) for current volatility
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr1_h[0] = high[0] - low[0]
    tr2_h[0] = np.abs(high[0] - close[0])
    tr3_h[0] = np.abs(low[0] - close[0])
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    atr_14_1h = pd.Series(tr_h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility filter: current 1h ATR > 1d ATR (elevated volatility regime)
    vol_filter = atr_14_1h > atr_14_1d_aligned
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20  # 20% position size
    
    # Start after warmup (need previous bar for Camarilla levels)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or filters not passed
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(r3[i]) or np.isnan(s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Apply session and volatility filters
        if not (session_filter[i] and vol_filter[i]):
            # Force flat when filters not passed
            signals[i] = 0.0
            position = 0
            continue
        
        # Long logic: price breaks above R3 + price above 4h EMA50 (uptrend)
        if close[i] > r3[i] and close[i] > ema_50_4h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + price below 4h EMA50 (downtrend)
        elif close[i] < s3[i] and close[i] < ema_50_4h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price returns to mean (close_prev) or reverses vs trend
        elif position == 1 and (close[i] < close_prev[i] or close[i] < ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > close_prev[i] or close[i] > ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0