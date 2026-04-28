#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe act as key support/resistance. 
Breakouts of these levels with 1-day trend filter (price > EMA34) and volume spikes 
capture strong momentum moves. Works in bull markets (breakouts continue up) and bear 
markets (breakdowns continue down) by following institutional levels. Targets 15-30 trades/year.
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
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Using previous day's OHLC to calculate today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla formula: range = prev_high - prev_low
    # R3 = prev_close + (range * 1.1/2)
    # S3 = prev_close - (range * 1.1/2)
    # R4 = prev_close + (range * 1.1)
    # S4 = prev_close - (range * 1.1)
    prev_range = prev_high - prev_low
    camarilla_r3 = prev_close + (prev_range * 1.1 / 2)
    camarilla_s3 = prev_close - (prev_range * 1.1 / 2)
    camarilla_r4 = prev_close + (prev_range * 1.1)
    camarilla_s4 = prev_close - (prev_range * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2x 20-period MA (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]
        breakdown_s3 = close[i] < s3_aligned[i]
        breakout_r4 = close[i] > r4_aligned[i]
        breakdown_s4 = close[i] < s4_aligned[i]
        
        # Volume confirmation (strong spike)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: breakout of R3/S3 with trend and volume
        # In strong trends, also allow R4/S4 breakouts
        long_entry = vol_confirm and ((uptrend and breakout_r3) or breakout_r4)
        short_entry = vol_confirm and ((downtrend and breakdown_s3) or breakdown_s4)
        
        # Exit logic: opposite breakdown or trend failure
        long_exit = breakdown_s3 or (not uptrend and breakdown_s4)
        short_exit = breakout_r3 or (not downtrend and breakout_r4)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0