#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d price action.
# Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following).
# Uses 1d price to calculate Camarilla levels: H-L range from previous day.
# R3 = Close + 1.1*(H-L)/4, S3 = Close - 1.1*(H-L)/4
# R4 = Close + 1.1*(H-L)/2, S4 = Close - 1.1*(H-L)/2
# Entry: Long when price crosses above S3 with rejection (close > open), short when crosses below R3 with rejection (close < open).
# Exit: Opposite signal or stop at R4/S4.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range for 6h.

name = "6h_camarilla_pivot_1d_meanrev_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    
    # Get 1d data for Camarilla calculation (previous day's H, L, C)
    df_1d = get_htf_data(prices, '1d')
    # Use previous day's values to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous day's data
    # R3 = C + 1.1*(H-L)/4, S3 = C - 1.1*(H-L)/4
    # R4 = C + 1.1*(H-L)/2, S4 = C - 1.1*(H-L)/2
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d / 4
    s3 = close_1d - 1.1 * range_1d / 4
    r4 = close_1d + 1.1 * range_1d / 2
    s4 = close_1d - 1.1 * range_1d / 2
    
    # Align to 6s timeframe (shifted by 1 day for previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for rejection
        # Skip if Camarilla data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Rejection conditions: close > open for bullish rejection, close < open for bearish rejection
        bullish_rejection = close[i] > open_[i]
        bearish_rejection = close[i] < open_[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches R4 (take profit) or bearish rejection at R3
            if (high[i] >= r4_aligned[i] or 
                (low[i] <= r3_aligned[i] and bearish_rejection)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S4 (take profit) or bullish rejection at S3
            if (low[i] <= s4_aligned[i] or 
                (high[i] >= s3_aligned[i] and bullish_rejection)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at S3/R3 with rejection
            # Long: price crosses above S3 with bullish rejection
            if (low[i] <= s3_aligned[i] and 
                close[i] > s3_aligned[i] and 
                bullish_rejection):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R3 with bearish rejection
            elif (high[i] >= r3_aligned[i] and 
                  close[i] < r3_aligned[i] and 
                  bearish_rejection):
                signals[i] = -0.25
                position = -1
    
    return signals