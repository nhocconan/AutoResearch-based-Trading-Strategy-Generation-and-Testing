#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above 20-bar Donchian high with volume > 1.5x 20-bar MA and 12h EMA34 rising.
Short when price breaks below 20-bar Donchian low with volume > 1.5x 20-bar MA and 12h EMA34 falling.
Exit when price touches the opposite Donchian band (middle) or reverses with volume.
Designed to capture strong trending moves with institutional volume in both bull and bear markets.
Volume confirmation ensures breakouts have conviction, reducing false signals.
Target: 20-35 trades/year per symbol to minimize fee drag.
"""

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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 20-bar Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Calculate 4h volume MA20 for confirmation
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 12h EMA34 slope (rising/falling)
    ema34_slope = np.diff(ema34_12h_aligned, prepend=ema34_12h_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-bar MA
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and rising 12h EMA34
            if (close[i] > donchian_high[i] and 
                volume_confirmed and 
                ema34_slope[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and falling 12h EMA34
            elif (close[i] < donchian_low[i] and 
                  volume_confirmed and 
                  ema34_slope[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian middle OR breaks below low with volume (reversal)
            if (close[i] <= donchian_middle[i] or 
                (close[i] < donchian_low[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian middle OR breaks above high with volume (reversal)
            if (close[i] >= donchian_middle[i] or 
                (close[i] > donchian_high[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0