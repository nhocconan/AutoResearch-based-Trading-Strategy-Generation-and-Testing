#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R(14) as momentum filter, 12h Donchian(20) breakout, and volume confirmation.
# Long when 1d Williams %R > -50, price breaks above 12h Donchian upper band, volume > 1.8x average.
# Short when 1d Williams %R < -50, price breaks below 12h Donchian lower band, volume > 1.8x average.
# Williams %R provides clearer momentum signals than RSI in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "12h_1dWilliamsR_12hDonchian_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data for Donchian bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_above_50 = williams_r > -50  # Above -50 = bullish momentum
    
    # 12h Donchian(20) bands
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Williams %R to 12h
    williams_above_50_aligned = align_htf_to_ltf(prices, df_1d, williams_above_50.astype(float))
    # Align 12h Donchian bands to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_above_50_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R > -50, price breaks above 12h Donchian upper band, volume spike
            if (williams_above_50_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: Williams %R < -50, price breaks below 12h Donchian lower band, volume spike
            elif (not williams_above_50_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: Williams %R flip, price breaks below Donchian lower band, or max 25 bars held
            if (not williams_above_50_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R flip, price breaks above Donchian upper band, or max 25 bars held
            if (williams_above_50_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals