#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Keltner calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate 1d EMA20 and ATR(10)
    ema_20_1d = pd.Series(prev_close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10_1d = pd.Series(np.maximum.reduce([
        prev_high_1d - prev_low_1d,
        np.abs(prev_high_1d - np.roll(prev_close_1d, 1)),
        np.abs(prev_low_1d - np.roll(prev_close_1d, 1))
    ])).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: EMA20 ± 1.5 * ATR10
    keltner_upper_1d = ema_20_1d + 1.5 * atr_10_1d
    keltner_lower_1d = ema_20_1d - 1.5 * atr_10_1d
    
    # Align Keltner channels to 4h
    keltner_upper_4h = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_4h = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: above 1.8x 8-period average (8*4h = 1.33 days)
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 8  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_4h[i]) or np.isnan(keltner_lower_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.8 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long breakout: price breaks above Keltner upper with daily uptrend
            if (close[i] > keltner_upper_4h[i] and 
                close[i] > ema_50_4h[i] and  # daily uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Keltner lower with daily downtrend
            elif (close[i] < keltner_lower_4h[i] and 
                  close[i] < ema_50_4h[i] and  # daily downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below EMA20 (mean reversion)
            if close[i] < keltner_upper_4h[i] - 0.5 * (keltner_upper_4h[i] - keltner_lower_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above EMA20 (mean reversion)
            if close[i] > keltner_lower_4h[i] + 0.5 * (keltner_upper_4h[i] - keltner_lower_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals