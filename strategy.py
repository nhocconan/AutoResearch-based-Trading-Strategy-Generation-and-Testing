#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d Supertrend filter and volume confirmation.
# Long when price breaks above R3 with Supertrend=1 (uptrend) and volume > 1.5x average.
# Short when price breaks below S3 with Supertrend=-1 (downtrend) and volume > 1.5x average.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Supertrend provides adaptive trend filtering that works in both bull and bear markets.
# Volume spike confirms institutional participation. Works in bull via upward breaks, bear via downward breaks.

name = "4h_Camarilla_R3_S3_Breakout_1dSupertrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (6 * 4h = ~24h)
    lookback = 6
    if n < lookback + 1:
        return np.zeros(n)
    
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Supertrend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend (10, 3.0) on 1d data
    def supertrend(high, low, close, period=10, multiplier=3.0):
        # True Range
        tr1 = pd.Series(high - low)
        tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
        tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        
        # Basic Upper and Lower Bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize Supertrend
        supertrend_vals = np.zeros_like(close)
        direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
        
        # First valid value
        supertrend_vals[period] = upper_band.iloc[period] if period < len(upper_band) else upper_band[-1]
        direction[period] = 1
        
        for i in range(period + 1, len(close)):
            # Current basic bands
            curr_upper = upper_band.iloc[i] if i < len(upper_band) else upper_band[-1]
            curr_lower = lower_band.iloc[i] if i < len(lower_band) else lower_band[-1]
            prev_supertrend = supertrend_vals[i-1]
            prev_direction = direction[i-1]
            
            # Supertrend logic
            if close[i] <= prev_supertrend:
                direction[i] = -1
                supertrend_vals[i] = curr_upper
            else:
                direction[i] = 1
                supertrend_vals[i] = curr_lower
            
            # Trend reversal checks
            if direction[i] == 1 and supertrend_vals[i] < curr_lower:
                supertrend_vals[i] = curr_lower
            if direction[i] == -1 and supertrend_vals[i] > curr_upper:
                supertrend_vals[i] = curr_upper
        
        return direction  # Returns 1 for uptrend, -1 for downtrend
    
    supertrend_1d = supertrend(high_1d, low_1d, close_1d, 10, 3.0)
    
    # Align 1d Supertrend to 4h timeframe (wait for 1d bar to close)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with Supertrend=1 (uptrend) and volume spike
            if (close[i] > r3[i] and 
                supertrend_1d_aligned[i] == 1 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with Supertrend=-1 (downtrend) and volume spike
            elif (close[i] < s3[i] and 
                  supertrend_1d_aligned[i] == -1 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal signal) OR Supertrend turns down
            if (close[i] < s3[i]) or (supertrend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal signal) OR Supertrend turns up
            if (close[i] > r3[i]) or (supertrend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals