#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_Filter
Hypothesis: On 6h timeframe, enter long when price closes above weekly R3 with close > weekly EMA34.
Enter short when price closes below weekly S3 with close < weekly EMA34.
Exit when price crosses weekly EMA34 (trend reversal).
Uses weekly timeframe for trend and structure, with 6t for execution.
Targets 12-30 trades/year (~50-120 total over 4 years) for low fee drag.
Works in bull markets via breakouts at R3/S3 and in bear via reversals at weekly EMA34.
"""

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Filter"
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
    
    # Load weekly data for Camarilla pivot calculation and EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Camarilla R3 and S3 levels
    r3 = weekly_pivot + weekly_range * 1.083 * 3  # R3 = pivot + range * 1.083 * 3
    s3 = weekly_pivot - weekly_range * 1.083 * 3  # S3 = pivot - range * 1.083 * 3
    
    # 1-week EMA34 for trend filter
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: 20-period moving average on 6t
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema1w_trend = ema34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R3 with close > weekly EMA34 and volume > 20MA
            if close[i] > r3_val and close[i] > ema1w_trend and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with close < weekly EMA34 and volume > 20MA
            elif close[i] < s3_val and close[i] < ema1w_trend and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA34 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA34 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals