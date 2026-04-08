#!/usr/bin/env python3
# 6h_weekly_pivot_volume_breakout_v1
# Hypothesis: Combines weekly pivot levels with 6-hour price breakouts and volume confirmation.
# Long when price breaks above weekly R3 with volume > 2x 20-period average.
# Short when price breaks below weekly S3 with volume > 2x 20-period average.
# Exit when price returns to weekly pivot (PP) or volume drops below average.
# Weekly pivots provide structural support/resistance that works in both bull and bear markets.
# Volume breakout confirms institutional participation.
# Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 2x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 2.0 * vol_ma[i]
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3
    # R3 = H + 2*(PP-L), S3 = L - 2*(H-PP)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = vol_ma_period  # Wait for volume MA to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly pivot or volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly pivot or volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] <= vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above weekly R3 with volume surge
            if (close[i] > r3_aligned[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S3 with volume surge
            elif (close[i] < s3_aligned[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals