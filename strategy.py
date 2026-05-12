#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price closes above weekly R3 with volume spike and price > 1w EMA50.
Enter short when price closes below weekly S3 with volume spike and price < 1w EMA50.
Exit when price crosses 1w EMA50 (trend reversal).
Uses weekly EMA50 for trend filter and weekly Camarilla R3/S3 for breakout levels.
Volume confirmation reduces false breakouts.
Designed to work in both bull and bear markets by following weekly trend.
Target: 15-30 trades/year for low fee drag.
"""

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Load weekly data for Camarilla pivot calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Camarilla R3 and S3 levels
    r3 = weekly_pivot + weekly_range * 1.250
    s3 = weekly_pivot - weekly_range * 1.250
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels and 1w EMA50 to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 24-period moving average (4 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema1w_trend = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R3 with price > 1w EMA50 and volume > 2x MA
            if close[i] > r3_val and close[i] > ema1w_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with price < 1w EMA50 and volume > 2x MA
            elif close[i] < s3_val and close[i] < ema1w_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1w EMA50 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1w EMA50 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals