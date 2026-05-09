#!/usr/bin/env python3
# Hypothesis: 12h Donchian breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above Donchian upper with EMA50 uptrend and volume > 2x average
# Short when price breaks below Donchian lower with EMA50 downtrend and volume > 2x average
# Exit when price retouches Donchian midline or reverses to opposite side
# Uses Donchian for trend structure, EMA for trend filter, volume for conviction
# Designed to capture breakouts in both trending and ranging markets with controlled frequency
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_Donchian_Breakout_1dEMA50_VolumeSpike"
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
    
    # Calculate 1d Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's Donchian channels (to avoid look-ahead)
    donch_high = df_1d['high'].rolling(window=20, min_periods=20).max().shift(1)
    donch_low = df_1d['low'].rolling(window=20, min_periods=20).min().shift(1)
    donch_mid = (donch_high + donch_low) / 2
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low.values)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid.values)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(donch_mid_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, EMA50 uptrend, volume spike
            if (close[i] > donch_high_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, EMA50 downtrend, volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches Donchian midline or reverses to lower band
            if (close[i] <= donch_mid_aligned[i]) or (close[i] < donch_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches Donchian midline or reverses to upper band
            if (close[i] >= donch_mid_aligned[i]) or (close[i] > donch_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals