#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels with volume confirmation.
Long when price touches S3 level with volume > 1.5x 20-period average and reverses upward.
Short when price touches R3 level with volume > 1.5x 20-period average and reverses downward.
Exit when price reaches opposite H3/L3 level or closes back inside H3/L3 range.
Designed for 15-25 trades/year to minimize fee drag while capturing mean reversion at extreme levels.
Works in both bull/bear markets as Camarilla adapts to volatility and volume filter avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's pivots
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_[range_ == 0] = 1e-10
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = pp + (range_ * 1.1 / 12)
    r2 = pp + (range_ * 1.1 / 6)
    r3 = pp + (range_ * 1.1 / 4)
    s1 = pp - (range_ * 1.1 / 12)
    s2 = pp - (range_ * 1.1 / 6)
    s3 = pp - (range_ * 1.1 / 4)
    
    # Align to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: 20-period average on 12h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 level (take profit) or closes back inside S3/S1
            if close[i] <= s1_aligned[i] or (close[i] >= s3_aligned[i] and close[i-1] < s3_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R1 level (take profit) or closes back inside R3/R1
            if close[i] >= r1_aligned[i] or (close[i] <= r3_aligned[i] and close[i-1] > r3_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches S3 and reverses up (close > S3 and prev close <= S3)
                if (close[i] <= s3_aligned[i] and close[i-1] > s3_aligned[i-1] and
                    close[i] > s3_aligned[i]):  # Reversal confirmation
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 and reverses down (close < R3 and prev close >= R3)
                elif (close[i] >= r3_aligned[i] and close[i-1] < r3_aligned[i-1] and
                      close[i] < r3_aligned[i]):  # Reversal confirmation
                    position = -1
                    signals[i] = -0.25
    
    return signals