#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend direction and volume confirmation.
- Donchian(20): Upper/lower bands from 20-period high/low on 12h chart.
- Trend Filter: Price > EMA34(1d) for long bias, Price < EMA34(1d) for short bias.
- Volume Confirmation: Current volume > 2.0 * 20-period average volume (strong spike).
- Entry: Long when price crosses above Donchian upper band AND price > EMA34(1d) AND volume spike.
         Short when price crosses below Donchian lower band AND price < EMA34(1d) AND volume spike.
- Exit: Opposite Donchian band touch (long exits when price touches lower band, short exits when price touches upper band).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d trend and requiring volume spikes for breakout validity.
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
    
    # Calculate Donchian(20) on 12h timeframe
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
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA34 for long bias, price < EMA34 for short bias
        long_bias = curr_close > ema34_1d_aligned[i]
        short_bias = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        upper_break = curr_high > highest_high[i-1] if i > 0 else False  # Break above previous upper band
        lower_break = curr_low < lowest_low[i-1] if i > 0 else False   # Break below previous lower band
        
        # Exit conditions: touch opposite Donchian band
        if position != 0:
            # Exit long: price touches or goes below lower Donchian band
            if position == 1:
                if curr_low <= lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price touches or goes above upper Donchian band
            elif position == -1:
                if curr_high >= highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper band AND long bias AND volume confirmation
            long_condition = upper_break and long_bias and volume_confirm
            
            # Short: break below lower band AND short bias AND volume confirmation
            short_condition = lower_break and short_bias and volume_confirm
            
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

name = "12h_Donchian20_Breakout_1dEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0