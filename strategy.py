#!/usr/bin/env python3
"""
Hypothesis: Daily Close Price Position within Weekly Donchian Channel + Volume Confirmation
Long when daily close is near weekly low (oversold in weekly trend) with volume spike.
Short when daily close is near weekly high (overbought in weekly trend) with volume spike.
Exit when price reverts to weekly middle or volume normalizes.
Designed for low trade frequency (~10-20/year) to capture mean-reversion in weekly extremes.
Works in both bull (buy dips) and bear (sell rallies) markets by trading weekly extremes.
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
    
    # Load weekly data for Donchian channel - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channel
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian high and low (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate volume spike (volume > 2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate price position within weekly Donchian channel (0 = at low, 1 = at high)
        channel_range = donchian_high_aligned[i] - donchian_low_aligned[i]
        if channel_range == 0:
            pos_in_channel = 0.5
        else:
            pos_in_channel = (close[i] - donchian_low_aligned[i]) / channel_range
        
        if position == 0:
            # Long: near weekly low (oversold) with volume spike
            if pos_in_channel < 0.2 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: near weekly high (overbought) with volume spike
            elif pos_in_channel > 0.8 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly middle or volume normalizes
                if pos_in_channel > 0.5 or not volume_spike[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly middle or volume normalizes
                if pos_in_channel < 0.5 or not volume_spike[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_WeeklyDonchianExtremes_Volume"
timeframe = "1d"
leverage = 1.0