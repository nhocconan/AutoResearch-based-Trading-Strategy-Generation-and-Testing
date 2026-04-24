#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d Supertrend trend filter + volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Supertrend (ATR=10, mult=3) for trend filter (price > Supertrend = uptrend, price < Supertrend = downtrend).
- Entry: Long when price breaks above Donchian upper band (20) AND price > 1d Supertrend AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian lower band (20) AND price < 1d Supertrend AND volume > 1.5 * 4h volume MA(20).
- Exit: Long exits when price breaks below Donchian lower band (10); Short exits when price breaks above Donchian upper band (10).
- Signal size: 0.25 discrete to balance capture and fee control.
- Donchian captures breakouts; Supertrend filters higher-timeframe trend; volume spike confirms conviction.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
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
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, mult=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = basic_lb[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = basic_ub[i]
            direction[i] = 1
        else:
            supertrend[i] = basic_lb[i]
            direction[i] = -1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Donchian channels on 4h (20-period for entry, 10-period for exit)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Get 4h data for volume MA(20)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 10)  # Donchian 20 needs 20, Supertrend needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(highest_high_10[i]) or np.isnan(lowest_low_10[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter from 1d Supertrend
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if uptrend and vol_confirm:
                # Long: price breaks above Donchian upper band (20)
                if curr_high > highest_high_20[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_confirm:
                # Short: price breaks below Donchian lower band (20)
                if curr_low < lowest_low_20[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian lower band (10)
            if curr_low < lowest_low_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian upper band (10)
            if curr_high > highest_high_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0