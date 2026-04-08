#!/usr/bin/env python3
# 1d_1w_donchian_breakout_volume
# Hypothesis: Trade weekly Donchian breakouts on daily timeframe with volume confirmation.
# Long when price breaks above weekly Donchian high (20-week) with volume surge.
# Short when price breaks below weekly Donchian low (20-week) with volume surge.
# Uses volume > 1.5x 20-day average for confirmation and ATR-based stops.
# Designed to capture major trends in both bull and bear markets with low trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-week high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling max/min for weekly data
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min_20)
    
    # ATR for volatility and stop (using daily data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low OR stoploss hit
            if close[i] < donchian_low[i] or close[i] < donchian_high[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high OR stoploss hit
            if close[i] > donchian_high[i] or close[i] > donchian_low[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly Donchian high with volume surge
            if close[i] > donchian_high[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with volume surge
            elif close[i] < donchian_low[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals