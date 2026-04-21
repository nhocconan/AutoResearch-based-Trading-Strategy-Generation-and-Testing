#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using daily pivot points (R1/S1) with 1h EMA50 trend filter and volume confirmation.
In uptrend (price > EMA50), buy breakouts above daily R1; in downtrend (price < EMA50), sell breakdowns below daily S1.
Daily R1/S1 provide institutional support/resistance with higher success rate than R2/S2.
1h EMA50 filters for stronger trend alignment; volume confirms breakout strength.
Works in bull markets (buy R1 breaks) and bear markets (sell S1 breaks). Target: 20-40 trades/year for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior day's H/L/C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = Pivot + (High - Low)
    r1_1d = pivot_1d + (high_1d - low_1d)
    # S1 = Pivot - (High - Low)
    s1_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily R1/S1 to 4h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Load 1h data ONCE before loop for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # 1h EMA50 for trend filter
    close_1h = df_1h['close'].values
    ema_50 = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above daily R1 + uptrend (price > EMA50) + volume spike
            if (price_close > r1_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S1 + downtrend (price < EMA50) + volume spike
            elif (price_close < s1_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA50 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DailyPivot_R1S1_1hEMA50_Volume"
timeframe = "4h"
leverage = 1.0