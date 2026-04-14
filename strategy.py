#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation
# - Long when price breaks above previous week's high with volume > 1.5x 20-day average
# - Short when price breaks below previous week's low with volume > 1.5x 20-day average
# - Exits on opposite breakout
# - Position size 0.25 to manage risk
# - Target: 30-100 trades over 4 years (7-25/year) to avoid fee drag
# - Weekly timeframe provides fewer, higher-quality signals suitable for 1d chart

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if volume MA is NaN
        if np.isnan(vol_ma[i]):
            continue
        
        # Get previous week's high/low for breakout levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        
        # Create arrays for alignment (constant values for the week)
        high_array = np.full(len(df_1w), prev_high)
        low_array = np.full(len(df_1w), prev_low)
        
        # Align to 1d timeframe
        high_1d = align_htf_to_ltf(prices, df_1w, high_array)[i]
        low_1d = align_htf_to_ltf(prices, df_1w, low_array)[i]
        
        if position == 0:
            # Long: Break above previous week's high with volume confirmation
            if close[i] > high_1d and volume[i] > vol_ma[i] * 1.5:
                position = 1
                signals[i] = position_size
            # Short: Break below previous week's low with volume confirmation
            elif close[i] < low_1d and volume[i] > vol_ma[i] * 1.5:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous week's low
            if close[i] < low_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous week's high
            if close[i] > high_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_DonchianBreakout_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0