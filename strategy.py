#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band with EMA50 uptrend and volume > 1.5x average
# Short when price breaks below Donchian lower band with EMA50 downtrend and volume > 1.5x average
# Exit when price retraces to Donchian midpoint or reverses to opposite band
# Uses Donchian channels for breakout structure, EMA for trend filter, volume for conviction
# Designed to capture breakouts with controlled frequency in both trending and ranging markets
# Target: 75-150 total trades over 4 years (19-38/year) with size 0.25

name = "4h_Donchian20_1dEMA50_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's Donchian bands (to avoid look-ahead)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    
    # Calculate Donchian channels
    upper_band = prev_high.rolling(window=20, min_periods=20).max()
    lower_band = prev_low.rolling(window=20, min_periods=20).min()
    middle_band = (upper_band + lower_band) / 2
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band.values)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band.values)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_band.values)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, EMA50 uptrend, volume confirmation
            if (close[i] > upper_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, EMA50 downtrend, volume confirmation
            elif (close[i] < lower_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to middle band or reverses to lower band
            if (close[i] <= middle_aligned[i]) or (close[i] < lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to middle band or reverses to upper band
            if (close[i] >= middle_aligned[i]) or (close[i] > upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals