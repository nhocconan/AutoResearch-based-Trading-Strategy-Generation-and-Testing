#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction and volume spike filter (>2.0x 20-period average).
- Donchian breakout: Long when close > highest high of past 20 bars, Short when close < lowest low of past 20 bars.
- Trend Filter: Price > EMA34(1d) for long bias, Price < EMA34(1d) for short bias.
- Volume Confirmation: Current volume > 2.0 * 20-period average volume (HTF 1d volume).
- Exit: Opposite Donchian breakout (long exits on close < lowest low of past 10 bars, short exits on close > highest high of past 10 bars).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Works in both bull and bear markets by aligning with 1d trend and using volatility-based breakouts with volume confirmation.
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
    
    # Calculate Donchian channels (20 for entry, 10 for exit)
    donchian_period = 20
    exit_period = 10
    
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    highest_high_exit = pd.Series(high).rolling(window=exit_period, min_periods=exit_period).max().values
    lowest_low_exit = pd.Series(low).rolling(window=exit_period, min_periods=exit_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 34)  # Need 20 for Donchian, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(highest_high_exit[i]) or np.isnan(lowest_low_exit[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price > EMA34 for long bias, price < EMA34 for short bias
        long_bias = curr_close > ema34_1d_aligned[i]
        short_bias = curr_close < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        breakout_long = curr_close > highest_high[i]
        breakout_short = curr_close < lowest_low[i]
        
        # Exit conditions: opposite Donchian (shorter period)
        exit_long = curr_close < lowest_low_exit[i]
        exit_short = curr_close > highest_high_exit[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below lower Donchian (10-period)
            if position == 1:
                if exit_long:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian (10-period)
            elif position == -1:
                if exit_short:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: Donchian breakout up AND long bias AND volume confirmation
            long_condition = breakout_long and long_bias and volume_confirm
            
            # Short: Donchian breakout down AND short bias AND volume confirmation
            short_condition = breakout_short and short_bias and volume_confirm
            
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

name = "4h_Donchian20_Breakout_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0