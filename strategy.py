#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly pivot levels with volume confirmation
# Weekly pivot points (R4/S4 for breakouts, R3/S3 for reversals) provide institutional support/resistance
# Price breaking above weekly R4 or below weekly S4 with volume > 1.5x 24-period average indicates institutional breakout
# Price rejecting at weekly R3 or S3 with volume confirmation indicates mean reversion opportunity
# Works in both bull/bear markets: breakouts capture trends, reversals capture pullbacks
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_WeeklyPivot_R3S4_VolumeBreakout_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot levels ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly pivot levels
    # Pivot = (previous high + previous low + previous close) / 3
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance and Support levels (weekly)
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align weekly levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: >1.5x 24-period average (2 days of 12h data)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R4 with volume confirmation
            if close[i] > r4_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly S4 with volume confirmation
            elif close[i] < s4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects weekly S3 with volume confirmation (bounce from support)
            elif close[i] < s3_aligned[i] and close[i] > s3_aligned[i] * 0.998 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects weekly R3 with volume confirmation (rejection from resistance)
            elif close[i] > r3_aligned[i] and close[i] < r3_aligned[i] * 1.002 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S3 (failed support) or reaches weekly R4 (take profit)
            if close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R3 (failed resistance) or reaches weekly S4 (take profit)
            if close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals