#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d Williams %R extreme + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Williams %R(14) for extreme overbought/oversold conditions.
- Entry: Long when price breaks above Donchian(20) high AND Williams %R < -80 (oversold) AND volume > 1.5 * 6h volume MA(20);
         Short when price breaks below Donchian(20) low AND Williams %R > -20 (overbought) AND volume > 1.5 * 6h volume MA(20).
- Exit: Long exits when price crosses below Donchian(20) low; Short exits when price crosses above Donchian(20) high.
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian provides trend-following structure; Williams %R extremes capture mean reversion in bear rallies/bull pullbacks;
  volume confirms breakout conviction. Works in bull (buying breakouts from oversold) and bear (selling breakdowns from overbought).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for highest high and lowest low
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R calculation
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Donchian(20) on 6h
    highest_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 6h data for volume MA(20)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Donchian needs 20, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(highest_high_6h[i]) or 
            np.isnan(lowest_low_6h[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Williams %R extremes: < -80 oversold, > -20 overbought
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if oversold and vol_confirm:
                # Long: price breaks above Donchian high
                if curr_high > highest_high_6h[i]:
                    signals[i] = 0.25
                    position = 1
            elif overbought and vol_confirm:
                # Short: price breaks below Donchian low
                if curr_low < lowest_low_6h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian low
            if curr_low < lowest_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian high
            if curr_high > highest_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWilliamsR_Extreme_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0