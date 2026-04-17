#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Donchian(20) breakout + volume confirmation + 12h EMA34 trend filter.
Long when price breaks above 12h Donchian upper band with volume confirmation and price > 12h EMA34 (uptrend).
Short when price breaks below 12h Donchian lower band with volume confirmation and price < 12h EMA34 (downtrend).
Exit when price returns to the 12h Donchian midpoint or reverses with volume.
Uses 12h timeframe for structure (reduces noise) and 4h for entry timing and volume confirmation.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Donchian channels provide robust support/resistance based on recent price extremes, effective in trending markets.
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
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    upper_12h = high_12h_series.rolling(window=20, min_periods=20).max().values
    lower_12h = low_12h_series.rolling(window=20, min_periods=20).min().values
    midpoint_12h = (upper_12h + lower_12h) / 2.0
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    midpoint_12h_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(midpoint_12h_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper band with volume and uptrend (price > EMA34)
            if (close[i] > upper_12h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band with volume and downtrend (price < EMA34)
            elif (close[i] < lower_12h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower band with volume (reversal)
            if (close[i] <= midpoint_12h_aligned[i] or 
                (close[i] < lower_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper band with volume (reversal)
            if (close[i] >= midpoint_12h_aligned[i] or 
                (close[i] > upper_12h_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian20_Breakout_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0