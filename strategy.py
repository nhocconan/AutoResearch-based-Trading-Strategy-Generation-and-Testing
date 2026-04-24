#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 75-150 total trades over 4 years (19-38/year).
- HTF: 12h for EMA50 trend direction (long bias when price > EMA50, short bias when price < EMA50).
- Donchian Channel(20): Upper = 20-period high, Lower = 20-period low.
- Volume Spike: Current 6h volume > 2.0 * 20-period average volume.
- Entry: Long when price breaks above Donchian Upper AND price > 12h EMA50 AND volume spike.
         Short when price breaks below Donchian Lower AND price < 12h EMA50 AND volume spike.
- Exit: Opposite Donchian breakout (long exits when price < Donchian Lower, short exits when price > Donchian Upper).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 12h trend and requiring volume confirmation for breakouts.
- Uses discrete signals to reduce churn and respects MTF data loading rules.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate Donchian Channel(20) on 6h timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50)  # Need 20 for Donchian, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_12h_aligned[i]
        short_bias = curr_close < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
        # Donchian breakout conditions
        broke_above_upper = curr_high > donchian_upper
        broke_below_lower = curr_low < donchian_lower
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian Lower
            if position == 1:
                if curr_low < donchian_lower:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian Upper
            elif position == -1:
                if curr_high > donchian_upper:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above Donchian Upper AND long bias AND volume confirmation
            long_condition = broke_above_upper and long_bias and volume_confirm
            
            # Short: break below Donchian Lower AND short bias AND volume confirmation
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

name = "6h_Donchian20_Breakout_12hEMA50Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0