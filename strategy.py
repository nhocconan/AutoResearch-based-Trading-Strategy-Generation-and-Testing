#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action relative to weekly pivot levels with volume confirmation
# Long when: price crosses above weekly R3 level with volume > 2x average
# Short when: price crosses below weekly S3 level with volume > 2x average
# Exit when: price returns to weekly pivot (PP) level or opposite crossover occurs
# Weekly pivot levels calculated from prior week's OHLC: PP=(H+L+C)/3, R3=H+2*(H-L), S3=L-2*(H-L)
# Designed to capture breakout momentum in both trending and ranging markets
# Target: 60-120 trades over 4 years by using weekly structure as filter

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels from prior week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r3 = weekly_high + 2 * (weekly_high - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe (shifted by 1 week for no look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Volume confirmation: volume > 2x 24-period average (4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Wait for volume MA to stabilize
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below weekly PP OR opposite crossover below S3
            if close[i] < pp_aligned[i] or (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above weekly PP OR opposite crossover above R3
            if close[i] > pp_aligned[i] or (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: price crosses R3/S3 with volume confirmation
            if volume[i] > volume_threshold[i]:
                # Long entry: price crosses above R3
                if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: price crosses below S3
                elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
    
    return signals