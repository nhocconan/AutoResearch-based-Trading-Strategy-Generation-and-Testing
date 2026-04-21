#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using daily pivot points (R3/S3) with 4h EMA50 trend filter and volume confirmation.
R3/S3 are stronger institutional levels than R1/S1/R2/S2, providing fewer but higher-quality breakouts.
In uptrend (price > EMA50), buy breakouts above daily R3; in downtrend (price < EMA50), sell breakdowns below daily S3.
EMA50 filters for trend alignment; volume confirms breakout strength with a higher threshold (2.0x avg volume).
This reduces trade frequency to target 15-25 trades/year, minimizing fee drag while capturing strong trending moves.
Works in bull markets (buy R3 breaks) and bear markets (sell S3 breaks).
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
    # R3 = Pivot + 2*(High - Low)
    r3_1d = pivot_1d + 2.0 * (high_1d - low_1d)
    # S3 = Pivot - 2*(High - Low)
    s3_1d = pivot_1d - 2.0 * (high_1d - low_1d)
    
    # Align daily R3/S3 to 4h timeframe (wait for daily bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 4h volume confirmation (volume spike > 2.0x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 2.0  # Higher volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above daily R3 + uptrend (price > EMA50) + volume spike
            if (price_close > r3_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily S3 + downtrend (price < EMA50) + volume spike
            elif (price_close < s3_aligned[i] and 
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

name = "4h_DailyPivot_R3S3_4hEMA50_Volume"
timeframe = "4h"
leverage = 1.0