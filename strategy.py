#!/usr/bin/env python3
"""
12h strategy using weekly pivot points (R3/S3) with 1d EMA34 trend filter and volume confirmation.
In uptrend (price > EMA34), buy breakouts above weekly R3; in downtrend (price < EMA34), sell breakdowns below weekly S3.
Weekly R3/S3 provide stronger institutional support/resistance, reducing false breakouts.
EMA34 filters for trend alignment; volume confirms breakout strength.
Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag.
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
    # R3 = Pivot + 2*(High - Low)
    r3_1w = pivot_1w + 2.0 * (high_1w - low_1w)
    # S3 = Pivot - 2*(High - Low)
    s3_1w = pivot_1w - 2.0 * (high_1w - low_1w)
    
    # Align weekly R3/S3 to 12h timeframe (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 12h volume confirmation (volume spike > 2.0x 30-period average)
    vol_ma_30 = pd.Series(prices['volume'].values).rolling(window=30, min_periods=30).mean().values
    vol_ratio = prices['volume'].values / vol_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above weekly R3 + uptrend (price > EMA34) + volume spike
            if (price_close > r3_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S3 + downtrend (price < EMA34) + volume spike
            elif (price_close < s3_aligned[i] and 
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

name = "12h_WeeklyPivot_R3S3_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0