#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using weekly pivot points (R2/S2) with 12h EMA50 trend filter and volume confirmation.
In uptrend (price > EMA50), buy breakouts above weekly R2; in downtrend (price < EMA50), sell breakdowns below weekly S2.
Weekly pivots provide stronger institutional support/resistance than daily, reducing false breakouts.
12h EMA50 filters for stronger trend alignment; volume confirms breakout strength.
Works in bull markets (buy R2 breaks) and bear markets (sell S2 breaks). Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's H/L/C)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R2 = Pivot + (High - Low)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # S2 = Pivot - (High - Low)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly R2/S2 to 4h timeframe (wait for weekly bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 4h volume confirmation (volume spike > 1.8x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Higher volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above weekly R2 + uptrend (price > EMA50) + volume spike
            if (price_close > r2_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S2 + downtrend (price < EMA50) + volume spike
            elif (price_close < s2_aligned[i] and 
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

name = "4h_WeeklyPivot_R2S2_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0