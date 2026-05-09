#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume spike
# Long when price breaks above 12h Donchian Upper with 1d EMA200 uptrend and volume > 2x average
# Short when price breaks below 12h Donchian Lower with 1d EMA200 downtrend and volume > 2x average
# Exit when price crosses 12h Donchian Middle or reverses to opposite Donchian band
# Uses Donchian for price channels, EMA200 for long-term trend, volume for conviction
# Designed to capture major breakouts with low frequency and high conviction
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Donchian_20_1dEMA200_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = high_20.values
    donchian_lower = low_20.values
    donchian_middle = ((high_20 + low_20) / 2).values
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian Upper, EMA200 uptrend, volume spike
            if (close[i] > donchian_upper[i] and 
                ema200_1d_aligned[i] > ema200_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian Lower, EMA200 downtrend, volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema200_1d_aligned[i] < ema200_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses Donchian Middle or reverses to Donchian Lower
            if (close[i] <= donchian_middle[i]) or (close[i] < donchian_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses Donchian Middle or reverses to Donchian Upper
            if (close[i] >= donchian_middle[i]) or (close[i] > donchian_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals