#!/usr/bin/env python3
# 12h_daily_camarilla_pullback_v1
# Hypothesis: 12h strategy using daily Camarilla pivot levels with pullback entries.
# Long: Price pulls back to daily R3 level after breaking above R4, with volume > 1.3x 20-period average.
# Short: Price pulls back to daily S3 level after breaking below S4, with volume > 1.3x 20-period average.
# Exit: Price returns to daily pivot point (PP) or breaks opposite S3/R3 level.
# Uses daily Camarilla for key support/resistance, 12h for execution with pullback confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + range_1d * 1.1 / 4.0
    r4 = close_1d + range_1d * 1.1 / 2.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    s4 = close_1d - range_1d * 1.1 / 2.0
    pp = pivot  # Pivot point
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    breakout_high = False  # Track if we've seen R4 breakout
    breakout_low = False   # Track if we've seen S4 breakout
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Track breakouts for pullback context
        if close[i] > r4_aligned[i]:
            breakout_high = True
        if close[i] < s4_aligned[i]:
            breakout_low = True
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot or breaks below S3
            if close[i] <= pp_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                breakout_high = False  # Reset breakout tracking
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot or breaks above R3
            if close[i] >= pp_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                breakout_low = False  # Reset breakout tracking
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback to R3 after R4 breakout (long)
            pullback_to_r3 = abs(close[i] - r3_aligned[i]) < (r4_aligned[i] - r3_aligned[i]) * 0.3
            bullish_setup = breakout_high and pullback_to_r3 and volume_confirmed
            
            # Check for pullback to S3 after S4 breakout (short)
            pullback_to_s3 = abs(close[i] - s3_aligned[i]) < (s3_aligned[i] - s4_aligned[i]) * 0.3
            bearish_setup = breakout_low and pullback_to_s3 and volume_confirmed
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals