#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 1d Donchian upper channel with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 1d Donchian lower channel with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 1d Donchian midpoint or reverses with volume.
Uses 1d timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Donchian channels provide dynamic support/resistance based on 20-period high/low, effective in both trending and ranging markets.
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    upper_20 = high_1d_series.rolling(window=20, min_periods=20).max().values
    lower_20 = low_1d_series.rolling(window=20, min_periods=20).min().values
    midpoint_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    midpoint_20_aligned = align_htf_to_ltf(prices, df_1d, midpoint_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(midpoint_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume and uptrend (price > EMA50)
            if (close[i] > upper_20_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume and downtrend (price < EMA50)
            elif (close[i] < lower_20_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower with volume (reversal)
            if (close[i] <= midpoint_20_aligned[i] or 
                (close[i] < lower_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper with volume (reversal)
            if (close[i] >= midpoint_20_aligned[i] or 
                (close[i] > upper_20_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Breakout_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0