#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using weekly pivot points (R1/S1) with 1d EMA trend filter and volume confirmation.
In uptrend (price > EMA34), buy breakouts above weekly R1; in downtrend (price < EMA34), sell breakdowns below weekly S1.
Weekly pivots provide strong institutional support/resistance; EMA34 filters for trend alignment; volume confirms breakout strength.
Works in bull markets (buy R1 breaks) and bear markets (sell S1 breaks). Target: 15-25 trades/year to minimize fee drag.
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
    # R1 = 2*Pivot - Low
    r1_1w = 2 * pivot_1w - low_1w
    # S1 = 2*Pivot - High
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly R1/S1 to 6h timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Load daily data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 6h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter
        
        if position == 0:
            # Enter long: price breaks above weekly R1 + uptrend (price > EMA34) + volume spike
            if (price_close > r1_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 + downtrend (price < EMA34) + volume spike
            elif (price_close < s1_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA34 in opposite direction)
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

name = "6h_WeeklyPivot_R1S1_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0