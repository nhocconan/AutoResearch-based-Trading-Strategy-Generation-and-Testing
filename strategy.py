#!/usr/bin/env python3
# 4h_1d_Donchian_Breakout_Volume_Trend_Filter
# Hypothesis: Breakout above Donchian upper/lower band on 4h timeframe with daily EMA34 trend filter.
# In uptrends (price > daily EMA34), long Donchian upper breakouts; in downtrends (price < daily EMA34), short Donchian lower breakouts.
# Added volume confirmation (current volume > 1.5x 20-period average) to filter false breakouts.
# Exit on opposite Donchian band break or trend reversal.
# Target: 20-40 trades/year to stay under 400 total trades over 4 years.

name = "4h_1d_Donchian_Breakout_Volume_Trend_Filter"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: uptrend + volume confirmation + break above Donchian high
            if close[i] > ema34_aligned[i] and volume_confirm and close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume confirmation + break below Donchian low
            elif close[i] < ema34_aligned[i] and volume_confirm and close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (reversal) or trend changes to downtrend
            if close[i] < donchian_low[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (reversal) or trend changes to uptrend
            if close[i] > donchian_high[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals