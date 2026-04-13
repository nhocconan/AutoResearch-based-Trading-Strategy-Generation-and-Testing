# 6h_1d_1w_Aggressive_Range_Bound
# Hypothesis: In strong weekly trends, price often consolidates between daily S3/R3.
# Enter long near daily S1 when weekly bullish and price touches S1 with volume expansion.
# Enter short near daily R1 when weekly bearish and price touches R1 with volume expansion.
# Exit when price reaches daily S3/R3 or weekly trend reverses.
# Works in bull/bear markets as it follows weekly trend direction.
# Target: 15-30 trades/year (60-120 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    C = close
    H = high
    L = low
    
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on daily
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    
    # Align all data to 6h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: weekly bullish, price touches S1, volume expansion
        long_condition = (low[i] <= S1_1d_aligned[i]) and weekly_bullish_aligned[i] > 0.5 and volume_expansion[i]
        
        # Short setup: weekly bearish, price touches R1, volume expansion
        short_condition = (high[i] >= R1_1d_aligned[i]) and weekly_bullish_aligned[i] < 0.5 and volume_expansion[i]
        
        # Exit conditions: price reaches S3/R3 or weekly trend reverses
        exit_long = (high[i] >= S3_1d_aligned[i]) or (weekly_bullish_aligned[i] < 0.5 and position == 1)
        exit_short = (low[i] <= R3_1d_aligned[i]) or (weekly_bullish_aligned[i] > 0.5 and position == -1)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_Aggressive_Range_Bound"
timeframe = "6h"
leverage = 1.0