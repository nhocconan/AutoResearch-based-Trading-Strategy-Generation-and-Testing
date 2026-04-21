#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Williams Alligator for trend direction and 1d Williams %R for mean-reversion entries.
Long when price is above Alligator teeth (green line) and %R crosses above -50 from below.
Short when price is below Alligator teeth and %R crosses below -50 from above.
Requires volume > 1.5x 20-period average to confirm momentum.
Exit when price crosses Alligator lips (red line) or %R reaches extreme (> -10 or < -90).
Designed for 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing swings in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for Williams Alligator (13,8,5 SMAs of median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    median_1w = (high_1w + low_1w) / 2
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMAs of median price
    jaw = pd.Series(median_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_1w).rolling(window=5, min_periods=5).mean().values
    
    # Load daily data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align weekly Alligator and daily Williams %R to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        wr_val = williams_r_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above teeth (bullish alignment) and WR crosses above -50 from below
            if (price_close > teeth_val and 
                wr_val > -50 and 
                # Check if previous WR was below -50 (crossing up)
                i > 0 and not np.isnan(williams_r_aligned[i-1]) and williams_r_aligned[i-1] <= -50 and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price below teeth (bearish alignment) and WR crosses below -50 from above
            elif (price_close < teeth_val and 
                  wr_val < -50 and 
                  # Check if previous WR was above -50 (crossing down)
                  i > 0 and not np.isnan(williams_r_aligned[i-1]) and williams_r_aligned[i-1] >= -50 and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses lips OR WR reaches extreme
            exit_signal = False
            
            if position == 1:
                # Long exit: price below lips OR WR < -10 (overbought)
                if price_close < lips_val or wr_val < -10:
                    exit_signal = True
            elif position == -1:
                # Short exit: price above lips OR WR > -90 (oversold)
                if price_close > lips_val or wr_val > -90:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_TR_WilliamsR_MR_Volume1.5"
timeframe = "1d"
leverage = 1.0