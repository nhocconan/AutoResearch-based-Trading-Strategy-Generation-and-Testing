#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R3, S3) from 1d data for mean-reversion entries,
# filtered by 1d EMA34 trend direction and volume confirmation. Enter long when price closes above R3 in an uptrend,
# enter short when price closes below S3 in a downtrend. Exit on close back below R3 (long) or above S3 (short).
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull (catch breakouts)
# and bear (catch breakdowns) with trend filter and volume confirmation.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for a given period.
    Returns R3, S3 levels.
    Formula:
    R3 = close + (high - low) * 1.1 / 4
    S3 = close - (high - low) * 1.1 / 4
    """
    range_hl = high - low
    r3 = close + range_hl * 1.1 / 4
    s3 = close - range_hl * 1.1 / 4
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3, S3 on daily data
    r3_1d, s3_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price closes above R3 in uptrend with volume confirmation
            if close[i] > r3_1d_aligned[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 in downtrend with volume confirmation
            elif close[i] < s3_1d_aligned[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes back below R3
            if close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above S3
            if close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals