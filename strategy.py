#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_alligator_elder_ray_v1
# Uses weekly Elder Ray Index (bull/bear power) from 1d data to determine trend direction,
# and Alligator (SMAs) on 1w data to filter for trending vs ranging markets.
# Long when bull power > 0, bear power < 0, and price > Alligator's teeth (green line) on weekly.
# Short when bear power < 0, bull power > 0, and price < Alligator's teeth on weekly.
# Uses 13/8/5 SMAs for Alligator jaws/teeth/lips.
# Designed for low trade frequency in both bull and bear markets by requiring trend alignment
# across multiple timeframes and avoiding choppy conditions via Alligator convergence.

name = "6h_1w_1d_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for Alligator (SMAs)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:  # need at least 13 for slowest SMA
        return np.zeros(n)
    
    # Calculate Elder Ray from daily data
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe (daily values update after daily bar closes)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Alligator from weekly data: SMAs of median price
    # Median price = (High + Low) / 2
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2.0
    
    # Alligator lines: Jaw (13-period), Teeth (8-period), Lips (5-period) SMAs
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align Alligator lines to 6h timeframe (weekly values update after weekly bar closes)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment check: lips > teeth > jaw = bullish alignment
        # lips < teeth < jaw = bearish alignment
        bullish_align = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_align = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Long conditions: bullish Elder Ray + bullish Alligator alignment
        if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and bullish_align and position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: bearish Elder Ray + bearish Alligator alignment
        elif (bear_power_aligned[i] < 0 and bull_power_aligned[i] > 0 and bearish_align and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit when Alligator lines converge (market becomes ranging)
        # Convergence: |lips - jaw| < small threshold relative to price
        elif position == 1 and abs(lips_aligned[i] - jaw_aligned[i]) < 0.001 * close[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and abs(lips_aligned[i] - jaw_aligned[i]) < 0.001 * close[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals