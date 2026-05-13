#!/usr/bin/env python3
"""
12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Price breaking above weekly Camarilla R3 or below S3 indicates strong momentum.
Confirmed by 1-week EMA50 trend filter and volume spike (volume > 1.5x 20-period average).
Works in bull markets (breakouts above R3) and bear markets (breakdowns below S3).
Designed for 12h timeframe to limit trades (target 50-150/4 years) and avoid fee drag.
"""

name = "12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: R3, S3
    # Formula: R3 = Close + 1.1 * (High - Low) * 1.1000, S3 = Close - 1.1 * (High - Low) * 1.1000
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    camarilla_r3 = weekly_close + 1.1 * (weekly_high - weekly_low) * 1.1000
    camarilla_s3 = weekly_close - 1.1 * (weekly_high - weekly_low) * 1.1000
    
    # Align Camarilla levels to 12h timeframe (wait for weekly bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Get 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above weekly R3 + volume spike + price above weekly EMA50
            if close[i] > r3_aligned[i] and vol_spike and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + volume spike + price below weekly EMA50
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly S3 or price breaks below EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly R3 or price breaks above EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals