#!/usr/bin/env python3
"""
6h_1d_Camarilla_Pivot_Breakout_V2
Hypothesis: Uses daily Camarilla pivot levels for breakout direction on 6h timeframe.
In ranging markets, price tends to revert from R3/S3 levels. In trending markets,
breakouts beyond R4/S4 levels with volume confirmation continue the trend.
Works in both bull and bear markets by adapting to volatility regimes.
Target: 12-30 trades/year on 6h (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Using previous day's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r4 = prev_close + (range_1d * 1.1 / 2)
    camarilla_r3 = prev_close + (range_1d * 1.1 / 4)
    camarilla_s3 = prev_close - (range_1d * 1.1 / 4)
    camarilla_s4 = prev_close - (range_1d * 1.1 / 2)
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean()
    volume_expansion_6h = volume_6h > (vol_ma_20_6h * 1.5)
    
    # Breakout conditions: close beyond R4/S4 with volume expansion
    breakout_up = (close_6h > camarilla_r4) & volume_expansion_6h
    breakout_down = (close_6h < camarilla_s4) & volume_expansion_6h
    
    # Reversion conditions: touch R3/S3 without breaking through
    touch_r3 = (high_6h >= camarilla_r3) & (close_6h < camarilla_r4)
    touch_s3 = (low_6h <= camarilla_s3) & (close_6h > camarilla_s3)
    
    # Align all signals to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    breakout_up_aligned = align_htf_to_ltf(prices, df_6h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_6h, breakout_down)
    touch_r3_aligned = align_htf_to_ltf(prices, df_6h, touch_r3)
    touch_s3_aligned = align_htf_to_ltf(prices, df_6h, touch_s3)
    
    # Session filter: 00:00-23:00 UTC (trade all hours for 6h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 0) & (hours <= 23)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(camarilla_r4_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]) or \
           np.isnan(touch_r3_aligned[i]) or \
           np.isnan(touch_s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trading logic
        if breakout_up_aligned[i]:
            # Breakout above R4: go long
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif breakout_down_aligned[i]:
            # Breakdown below S4: go short
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        elif touch_r3_aligned[i] and position != -1:
            # Touch R3: short reversion (only if not already short)
            position = -1
            signals[i] = -position_size
        elif touch_s3_aligned[i] and position != 1:
            # Touch S3: long reversion (only if not already long)
            position = 1
            signals[i] = position_size
        elif position == 1:
            # Hold long position
            signals[i] = position_size
        elif position == -1:
            # Hold short position
            signals[i] = -position_size
        else:
            # Flat
            signals[i] = 0.0
    
    return signals

name = "6h_1d_Camarilla_Pivot_Breakout_V2"
timeframe = "6h"
leverage = 1.0