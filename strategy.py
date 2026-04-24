#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend direction and 1d for volume spike filter.
- Donchian Channel: Upper = 20-period high, Lower = 20-period low.
- Trend Filter: Price > EMA34(1w) for long bias, Price < EMA34(1w) for short bias.
- Volume Confirmation: Current volume > 2.0 * 20-period average volume (1d).
- Entry: Long when close breaks above Upper AND long bias AND volume confirmation.
         Short when close breaks below Lower AND short bias AND volume confirmation.
- Exit: Opposite Donchian breakout (long exits when close < Lower, short exits when close > Upper).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1w trend and filtering with volume spikes.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian Channel(20) on 12h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 34)  # Need 20 for Donchian, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        
        # Trend filter: price > EMA34 for long bias, price < EMA34 for short bias
        long_bias = curr_close > ema34_1w_aligned[i]
        short_bias = curr_close < ema34_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper
        broke_below_lower = curr_close < lower
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: close breaks below lower band
            if position == 1:
                if curr_close < lower:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper band
            elif position == -1:
                if curr_close > upper:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper AND long bias AND volume confirmation
            long_condition = broke_above_upper and long_bias and volume_confirm
            
            # Short: break below lower AND short bias AND volume confirmation
            short_condition = broke_below_lower and short_bias and volume_confirm
            
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

name = "12h_Donchian20_Breakout_1wEMA34Trend_1dVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0