#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R3/S3) from daily data as support/resistance.
# Enter long when price breaks above R3 with volume confirmation and 1d uptrend.
# Enter short when price breaks below S3 with volume confirmation and 1d downtrend.
# Exit on price reversion to the daily pivot point (PP).
# Designed for low frequency (20-50 trades/year) to avoid fee drag.
# Works in bull markets (breakouts) and bear markets (breakdowns) with trend filter and volume confirmation.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels.
    Returns: R3, R2, R1, PP, S1, S2, S3
    """
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    PP = typical
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    R2 = close + range_val * 1.1 / 6
    S2 = close - range_val * 1.1 / 6
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    
    return R3, R2, R1, PP, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R3_1d = np.zeros(len(close_1d))
    S3_1d = np.zeros(len(close_1d))
    PP_1d = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        R3, _, _, PP, _, _, S3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        R3_1d[i] = R3
        S3_1d[i] = S3
        PP_1d[i] = PP
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(PP_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
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
            # LONG: Price breaks above R3 with volume confirmation and uptrend
            if close[i] > R3_1d_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume confirmation and downtrend
            elif close[i] < S3_1d_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to or below daily pivot (PP)
            if close[i] <= PP_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or above daily pivot (PP)
            if close[i] >= PP_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals