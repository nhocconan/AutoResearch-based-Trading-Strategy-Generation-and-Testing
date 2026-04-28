#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Alligator for trend filter with 1w Camarilla R3/S3 breakout and volume confirmation.
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) provides strong trend detection in both bull/bear markets.
# Breakout at 1w Camarilla R3/S3 levels (moderate levels = balanced frequency).
# Volume spike (>2.0x 28-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.

name = "12h_WilliamsAlligator_1wCamarillaR3S3_VolumeSpike_v1"
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
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator (Smoothed Medians)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Jaw (13-period smoothed median, 8 bars ahead)
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period smoothed median, 5 bars ahead)
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period smoothed median, 3 bars ahead)
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1w Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w_prev) / 3.0
    # Range = H - L
    range_1w = high_1w - low_1w
    # Camarilla levels (R3/S3 provide good breakout structure)
    R3 = pivot + range_1w * 1.1 / 4.0
    S3 = pivot - range_1w * 1.1 / 4.0
    
    # Align to 12h timeframe (use previous 1w bar's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Calculate 12h volume spike: >2.0x 28-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_28 = volume_series.rolling(window=28, min_periods=28).mean().values
    volume_spike = volume > 2.0 * volume_ma_28
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(volume_ma_28[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend: Mouth open (Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend)
        uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S3_aligned[i] or not uptrend
        short_exit = close[i] > R3_aligned[i] or not downtrend
        
        # Handle entries and exits
        if long_breakout and uptrend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals