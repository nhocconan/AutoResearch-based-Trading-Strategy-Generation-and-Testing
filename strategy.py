#!/usr/bin/env python3
# 4h_Williams_Alligator_Trend_Confirm_Volume
# Hypothesis: Williams Alligator (Jaw=13, Teeth=8, Lips=5) with teeth/lips crossing for trend direction,
# confirmed by price above/below jaw and volume spike. Works in bull/bear by following Alligator's
# alignment: bullish when teeth>lips and price>jaw, bearish when teeth<lips and price<jaw.
# Target: 20-40 trades/year to avoid fee drag. Uses Williams Alligator as primary trend filter.

name = "4h_Williams_Alligator_Trend_Confirm_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Williams Alligator components
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (SMMA)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price = (high_1d + low_1d) / 2
    
    # Jaw (13-period SMMA of median)
    jaw = np.full_like(median_price, np.nan)
    for i in range(13, len(median_price)):
        jaw[i] = np.mean(median_price[i-12:i+1])
    
    # Teeth (8-period SMMA of median)
    teeth = np.full_like(median_price, np.nan)
    for i in range(8, len(median_price)):
        teeth[i] = np.mean(median_price[i-7:i+1])
    
    # Lips (5-period SMMA of median)
    lips = np.full_like(median_price, np.nan)
    for i in range(5, len(median_price)):
        lips[i] = np.mean(median_price[i-4:i+1])
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all Alligator lines (13)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: teeth > lips and price > jaw
            if teeth_aligned[i] > lips_aligned[i] and close[i] > jaw_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Bearish alignment: teeth < lips and price < jaw
            elif teeth_aligned[i] < lips_aligned[i] and close[i] < jaw_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment or price below jaw
            if teeth_aligned[i] < lips_aligned[i] or close[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment or price above jaw
            if teeth_aligned[i] > lips_aligned[i] or close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals