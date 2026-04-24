#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend direction and volume spike filter.
- Donchian(20): Upper = 20-bar high, Lower = 20-bar low.
- Trend Filter: Price > EMA34(1d) for long bias, Price < EMA34(1d) for short bias.
- Volume Confirmation: Current volume > 1.8 * 20-period average volume.
- Entry: Long when close breaks above Upper AND long bias AND volume confirmation.
         Short when close breaks below Lower AND short bias AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits on Lower break, short exits on Upper break).
- Signal size: 0.30 discrete to balance return and fee drag.
- Works in both bull and bear markets by aligning with 1d trend and requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34)  # Need 20 for Donchian, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price > EMA34 for long bias, price < EMA34 for short bias
        long_bias = curr_close > ema34_1d_aligned[i]
        short_bias = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average volume
        volume_confirm = curr_volume > 1.8 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakouts
        upper_break = curr_close > highest_high[i-1] if i > 0 else False
        lower_break = curr_close < lowest_low[i-1] if i > 0 else False
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian Lower
            if position == 1:
                if curr_close < lowest_low[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian Upper
            elif position == -1:
                if curr_close > highest_high[i-1]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: close breaks above Upper AND long bias AND volume confirmation
            long_condition = upper_break and long_bias and volume_confirm
            
            # Short: close breaks below Lower AND short bias AND volume confirmation
            short_condition = lower_break and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0