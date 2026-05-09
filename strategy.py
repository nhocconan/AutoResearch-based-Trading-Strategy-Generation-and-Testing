#!/usr/bin/env python3
# Hypothesis: 1d price closes beyond 1-week Donchian channels with volume confirmation
# Long when price closes above 1-week Donchian upper band (20-period) and volume > 1.5x 20-day average
# Short when price closes below 1-week Donchian lower band (20-period) and volume > 1.5x 20-day average
# Exit when price closes back inside the 1-week Donchian channel
# Position size: 0.25 (25% of capital) to limit drawdown in volatile markets
# Designed to capture strong weekly trends while avoiding false breakouts in ranging markets
# Works in both bull and bear markets by following the dominant weekly trend

name = "1d_WeeklyDonchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Get 1-week data for Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels: 20-period high and low
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 1-week Donchian channels to daily timeframe (waits for weekly close)
    upper_band = align_htf_to_ltf(prices, df_1w, high_20)
    lower_band = align_htf_to_ltf(prices, df_1w, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price closes above weekly upper band + volume spike
            if (close[i] > upper_band[i]) and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below weekly lower band + volume spike
            elif (close[i] < lower_band[i]) and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes back inside weekly Donchian channel
            if close[i] <= upper_band[i]:  # Exit when not above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes back inside weekly Donchian channel
            if close[i] >= lower_band[i]:  # Exit when not below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals