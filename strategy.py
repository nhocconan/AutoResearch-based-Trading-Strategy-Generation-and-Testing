#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above upper Donchian band with EMA50 uptrend and volume > 1.5x average
# Short when price breaks below lower Donchian band with EMA50 downtrend and volume > 1.5x average
# Exit when price crosses below/above the middle Donchian band (20-period average)
# Uses price channel breakouts for trend continuation, EMA for trend filter, volume for conviction
# Designed to capture sustained moves in both bull and bear markets with low frequency
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "12h_Donchian_20_1dEMA50_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's high/low for Donchian calculation (using 20-period lookback)
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min()
    mid_20 = (high_20 + low_20) / 2
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_20.values)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_20.values)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20.values)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(mid_20_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band, EMA50 uptrend, volume spike
            if (close[i] > upper_20_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band, EMA50 downtrend, volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle Donchian band
            if close[i] < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle Donchian band
            if close[i] > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals