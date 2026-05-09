#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA21 trend filter and volume confirmation
# Long when price breaks above upper band with EMA21 uptrend and volume > 1.5x average
# Short when price breaks below lower band with EMA21 downtrend and volume > 1.5x average
# Exit when price crosses the 20-period EMA (middle band) or reverses to opposite band
# Uses Donchian channels for breakout structure, EMA for trend, volume for conviction
# Designed to capture sustained moves in trending markets with controlled frequency
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_Donchian20_EMA21_Trend_Volume"
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
    
    # Calculate 12h Donchian channels (20-period high/low)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate upper and lower bands (20-period high/low)
    upper = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate middle band (20-period EMA)
    middle = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle)
    
    # Calculate 12h EMA21 for trend filter
    ema21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(ema21_12h_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, EMA21 uptrend, volume confirmation
            if (close[i] > upper_aligned[i] and 
                ema21_12h_aligned[i] > ema21_12h_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, EMA21 downtrend, volume confirmation
            elif (close[i] < lower_aligned[i] and 
                  ema21_12h_aligned[i] < ema21_12h_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band or reverses to lower band
            if (close[i] < middle_aligned[i]) or (close[i] < lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band or reverses to upper band
            if (close[i] > middle_aligned[i]) or (close[i] > upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals