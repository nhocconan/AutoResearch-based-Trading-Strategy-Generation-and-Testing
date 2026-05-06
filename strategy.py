#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1w Supertrend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND 1w Supertrend is bullish AND volume > 1.3 * avg_volume(20) on 6h
# Short when 1d Williams %R > -20 (overbought) AND 1w Supertrend is bearish AND volume > 1.3 * avg_volume(20) on 6h
# Exit when 1d Williams %R crosses back through -50 (mean reversion)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies exhaustion points that work in both bull and bear markets
# 1w Supertrend ensures we trade with the dominant weekly trend direction
# Volume confirmation (1.3x) validates the exhaustion move while limiting overtrading

name = "6h_1dWilliamsR_1wSupertrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    williams_r_1d[highest_high_1d == lowest_low_1d] = -50  # avoid division by zero
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1w data ONCE before loop for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3.0)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend upper and lower bands
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    # Initialize Supertrend
    supertrend_1w = np.full_like(close_1w, np.nan)
    direction_1w = np.full_like(close_1w, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    if len(close_1w) >= 10:
        supertrend_1w[0] = upper_band_1w[0]
        direction_1w[0] = 1  # start with uptrend assumption
    
    # Calculate Supertrend iteratively
    for i in range(1, len(close_1w)):
        if supertrend_1w[i-1] == upper_band_1w[i-1]:
            # previously in uptrend
            if close_1w[i] <= upper_band_1w[i]:
                supertrend_1w[i] = upper_band_1w[i]
                direction_1w[i] = 1
            else:
                supertrend_1w[i] = lower_band_1w[i]
                direction_1w[i] = -1
        else:
            # previously in downtrend
            if close_1w[i] >= lower_band_1w[i]:
                supertrend_1w[i] = lower_band_1w[i]
                direction_1w[i] = -1
            else:
                supertrend_1w[i] = upper_band_1w[i]
                direction_1w[i] = 1
    
    # Align 1w Supertrend direction to 6h timeframe (wait for completed 1w bar)
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), Supertrend bullish, volume confirmation, in session
            if (williams_r_aligned[i] < -80 and 
                supertrend_direction_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), Supertrend bearish, volume confirmation, in session
            elif (williams_r_aligned[i] > -20 and 
                  supertrend_direction_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals