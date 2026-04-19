#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels with breakout/continuation logic.
# Uses weekly Pivot Points (PP, R1/S1, R2/S2, R3/S3) to identify key levels.
# Enters long when price breaks above weekly R2 with volume confirmation.
# Enters short when price breaks below weekly S2 with volume confirmation.
# Exits when price returns to weekly PP or reverses at R3/S3.
# Weekly pivot provides structural support/resistance that works in both bull and bear markets.
# Targets 15-30 trades/year (60-120 total over 4 years) with strict breakout conditions.
name = "6h_WeeklyPivot_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC) - optional filter for liquidity
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get weekly data for Pivot Points (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points
    # PP = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*PP - Low
    r1_1w = 2 * pp_1w - low_1w
    # S1 = 2*PP - High
    s1_1w = 2 * pp_1w - high_1w
    # R2 = PP + (High - Low)
    r2_1w = pp_1w + (high_1w - low_1w)
    # S2 = PP - (High - Low)
    s2_1w = pp_1w - (high_1w - low_1w)
    # R3 = High + 2*(PP - Low)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    # S3 = Low - 2*(High - PP)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume filter: volume > 1.8 * 20-period average to avoid false breakouts
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for weekly pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R2 with volume confirmation
            if (close[i] > r2_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 with volume confirmation
            elif (close[i] < s2_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to weekly PP or reverses at R3
            if close[i] <= pp_1w_aligned[i] or close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to weekly PP or reverses at S3
            if close[i] >= pp_1w_aligned[i] or close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals