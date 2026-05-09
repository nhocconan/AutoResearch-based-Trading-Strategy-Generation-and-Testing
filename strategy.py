#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator + Elder Ray Power Index with 1d trend filter
# Long when green line > red line (bullish alignment) and bull power > 0 with 1d EMA50 uptrend
# Short when red line > green line (bearish alignment) and bear power < 0 with 1d EMA50 downtrend
# Uses Williams Alligator for trend identification and Elder Ray for bull/bear power
# Designed to capture sustained trends with low frequency in both bull and bear markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "4h_Alligator_ElderRay_PowerTrend"
timeframe = "4h"
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
    
    # Calculate Williams Alligator (13,8,5 SMAs of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # Smoothed (13,8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # Smoothed (8,5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values  # Smoothed (5,3)
    
    # Calculate Elder Ray Power Index (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicator calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment (lips > teeth > jaw) and bull power > 0 with 1d EMA50 uptrend
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                bull_power[i] > 0 and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (jaw > teeth > lips) and bear power < 0 with 1d EMA50 downtrend
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and
                  bear_power[i] < 0 and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or bull power <= 0
            if (jaw[i] > teeth[i] or teeth[i] > lips[i] or bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or bear power >= 0
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals