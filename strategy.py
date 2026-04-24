#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Donchian(20) and volume MA(20), 1w for pivot-based trend filter.
- Entry: Long when price breaks above 6h Donchian(20) high AND weekly pivot shows bullish bias (price > weekly R1) AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below 6h Donchian(20) low AND weekly pivot shows bearish bias (price < weekly S1) AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when price crosses below 6h Donchian(20) low; Short exits when price crosses above 6h Donchian(20) high.
- Signal size: 0.25 discrete to control fee drag.
- Uses Donchian breakouts for structure, weekly pivots for higher-timeframe bias, and volume confirmation to avoid false breakouts.
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
    
    # Get 1d data for Donchian(20) and volume MA(20)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one completed weekly candle
        return np.zeros(n)
    
    # Calculate weekly pivot levels from prior 1w OHLC
    # Standard pivot: P = (high + low + close) / 3
    # R1 = 2*P - low, S1 = 2*P - high
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_R1 = 2 * weekly_pivot - low_1w
    weekly_S1 = 2 * weekly_pivot - high_1w
    
    # Align 1d indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align 1w indicators to 6h timeframe
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian and volume MA both need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(weekly_R1_aligned[i]) or 
            np.isnan(weekly_S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above 1d Donchian high AND price > weekly R1 (bullish bias)
                if curr_high > donchian_high_aligned[i] and curr_close > weekly_R1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below 1d Donchian low AND price < weekly S1 (bearish bias)
                elif curr_low < donchian_low_aligned[i] and curr_close < weekly_S1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below 1d Donchian low
            if curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above 1d Donchian high
            if curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dVolMA20_1wPivotR1S1_Breakout_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0