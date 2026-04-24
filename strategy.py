#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Donchian(20): Upper/lower bands from 20-period high/low.
- Trend Filter: Price > EMA50(1d) for long bias, Price < EMA50(1d) for short bias.
- Volume Confirmation: Current volume > 1.5 * 20-period average volume.
- Entry: Long when close breaks above Donchian upper AND price > EMA50 AND volume confirmation.
         Short when close breaks below Donchian lower AND price < EMA50 AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits when close < Donchian lower, short exits when close > Donchian upper).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d trend and filtering breakouts with volume.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian(20) on 4h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_1d_aligned[i]
        short_bias = curr_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        upper_breakout = curr_close > highest_high[i]
        lower_breakout = curr_close < lowest_low[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian lower
            if position == 1:
                if lower_breakout:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper
            elif position == -1:
                if upper_breakout:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: close breaks above Donchian upper AND long bias AND volume confirmation
            long_condition = upper_breakout and long_bias and volume_confirm
            
            # Short: close breaks below Donchian lower AND short bias AND volume confirmation
            short_condition = lower_breakout and short_bias and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0