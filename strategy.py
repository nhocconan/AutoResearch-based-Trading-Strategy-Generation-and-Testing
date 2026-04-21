#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using daily pivot points (R2/S2) with 4h EMA55 trend filter and volume confirmation.
In uptrend (price > EMA55), buy breakouts above daily R2; in downtrend (price < EMA55), sell breakdowns below daily S2.
Daily R2/S2 provide stronger institutional support/resistance than R1/S1, reducing false breakouts.
EMA55 filters for stronger trend alignment; volume confirms breakout strength.
Works in bull markets (buy R2 breaks) and bear markets (sell S2 breaks). Target: 15-37 trades/year for 1h timeframe.
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
    # R2 = Pivot + (High - Low)
    r2_1d = pivot_1d + (high_1d - low_1d)
    # S2 = Pivot - (High - Low)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily R2/S2 to 1h timeframe (wait for daily bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA55 for trend filter (slower, more reliable)
    close_4h = df_4h['close'].values
    ema_55 = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema_55_aligned = align_htf_to_ltf(prices, df_4h, ema_55)
    
    # 1h volume confirmation (volume spike > 1.8x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_55_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_55_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Higher volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above daily R2 + uptrend (price > EMA55) + volume spike
            if (price_close > r2_aligned[i] and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below daily S2 + downtrend (price < EMA55) + volume spike
            elif (price_close < s2_aligned[i] and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA55 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_DailyPivot_R2S2_4hEMA55_Volume"
timeframe = "1h"
leverage = 1.0