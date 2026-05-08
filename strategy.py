#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with daily trend filter and volume spike
# Uses daily Camarilla levels (R3/S3) for breakout entries, daily EMA34 for trend filter,
# and volume > 1.5x 20-period average for confirmation. Trades in direction of daily trend.
# Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla levels, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_range = prev_high - prev_low
    
    # Camarilla levels (R3, S3) based on previous day
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1d > np.roll(ema_34_1d, 1)
    trend_up = np.where(np.isnan(trend_up), False, trend_up)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d > (vol_avg_20 * 1.5)
    vol_spike = np.where(np.isnan(vol_spike), False, vol_spike)
    
    # Align daily indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3, daily trend up, volume spike
            if close[i] > r3_aligned[i] and trend_up_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S3, daily trend down, volume spike
            elif close[i] < s3_aligned[i] and not trend_up_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla range (between S3 and R3) or trend fails
            if close[i] < r3_aligned[i] and close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif not trend_up_aligned[i]:  # Trend failed
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Camarilla range or trend fails
            if close[i] > s3_aligned[i] and close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif trend_up_aligned[i]:  # Trend failed (now up)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals