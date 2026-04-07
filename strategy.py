#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v3
Hypothesis: Daily Camarilla pivot levels (S3/R3, S4/R4) act as strong support/resistance zones.
Price respects these levels with volume confirmation, offering mean-reversion bounces in range
and breakout continuation in trends. Uses 12h candles for lower frequency to reduce trade count
and avoid fee drag. Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v3"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    R4 = prev_close + 1.5 * (prev_high - prev_low)
    R3 = prev_close + 1.0 * (prev_high - prev_low)
    R2 = prev_close + 0.5 * (prev_high - prev_low)
    R1 = prev_close + 0.25 * (prev_high - prev_low)
    S1 = prev_close - 0.25 * (prev_high - prev_low)
    S2 = prev_close - 0.5 * (prev_high - prev_low)
    S3 = prev_close - 1.0 * (prev_high - prev_low)
    S4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align to 12h timeframe (shifted by 1 day for lookback)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any level or volume MA is not ready
        if (np.isnan(R4_12h[i]) or np.isnan(R3_12h[i]) or np.isnan(R2_12h[i]) or np.isnan(R1_12h[i]) or
            np.isnan(S1_12h[i]) or np.isnan(S2_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(S4_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S2 (strong support broken)
            if close[i] < S2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R2 (strong resistance broken)
            if close[i] > R2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price bounces from S3/S4 with volume (bullish reversal)
            # OR breaks above R3 with volume (bullish breakout)
            long_breakout = (close[i] > R3_12h[i] and close[i-1] <= R3_12h[i-1])
            long_bounce_s3 = (close[i] > S3_12h[i] and close[i-1] <= S3_12h[i-1])
            long_bounce_s4 = (close[i] > S4_12h[i] and close[i-1] <= S4_12h[i-1])
            
            if (long_breakout or long_bounce_s3 or long_bounce_s4) and close[i] < R2_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price rejects from R3/R4 with volume (bearish reversal)
            # OR breaks below S3 with volume (bearish breakdown)
            short_breakdown = (close[i] < S3_12h[i] and close[i-1] >= S3_12h[i-1])
            short_reject_r3 = (close[i] < R3_12h[i] and close[i-1] >= R3_12h[i-1])
            short_reject_r4 = (close[i] < R4_12h[i] and close[i-1] >= R4_12h[i-1])
            
            if (short_breakdown or short_reject_r3 or short_reject_r4) and close[i] > S2_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals