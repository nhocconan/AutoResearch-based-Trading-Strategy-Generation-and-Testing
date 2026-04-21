#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using daily pivot points (R1/S1) with 4h EMA trend filter and volume confirmation.
In uptrend (price > EMA21), buy breakouts above daily R1; in downtrend (price < EMA21), sell breakdowns below daily S1.
Daily pivots provide strong institutional support/resistance; EMA21 filters for trend alignment; volume confirms breakout strength.
Works in bull markets (buy R1 breaks) and bear markets (sell S1 breaks). Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    # R1 = 2*Pivot - Low
    r1_1d = 2 * pivot_1d - low_1d
    # S1 = 2*Pivot - High
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily R1/S1 to 4h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_21_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above daily R1 + uptrend (price > EMA21) + volume spike
            if (price_close > r1_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S1 + downtrend (price < EMA21) + volume spike
            elif (price_close < s1_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA21 in opposite direction)
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

name = "4h_DailyPivot_R1S1_4hEMA21_Volume"
timeframe = "4h"
leverage = 1.0