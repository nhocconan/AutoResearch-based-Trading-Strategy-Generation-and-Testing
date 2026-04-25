#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d Supertrend(10,3) trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Supertrend trend direction and Donchian channel calculation from prior day.
- Donchian Channel: Upper/lower bands from 20-period high/low of prior 1d OHLC.
- Trend Filter: 1d Supertrend must align with breakout direction (Supertrend = uptrend for long, downtrend for short).
- Volume Filter: Current 4h volume > 1.8 * 20-period average 4h volume to confirm strong momentum.
- Entry: Long when close > Upper Band AND Supertrend uptrend AND volume spike.
         Short when close < Lower Band AND Supertrend downtrend AND volume spike.
- Exit: Opposite Donchian break (long exits when close < Lower Band, short exits when close > Upper Band).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with daily trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
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
    
    # Calculate 1d Supertrend for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient data for Supertrend
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate 1d Donchian channels (using prior day data to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    
    # Donchian Channel: 20-period high/low
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        supertrend_level = supertrend_aligned[i]
        supertrend_dir = direction_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_band
        broke_below_lower = curr_close < lower_band
        
        # Trend alignment conditions
        uptrend = supertrend_dir == 1
        downtrend = supertrend_dir == -1
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower band
            if position == 1:
                if curr_close < lower_band:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper band
            elif position == -1:
                if curr_close > upper_band:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper band AND uptrend AND volume spike
            long_condition = broke_above_upper and uptrend and volume_spike
            
            # Short: break below lower band AND downtrend AND volume spike
            short_condition = broke_below_lower and downtrend and volume_spike
            
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

name = "4h_Donchian20_Breakout_1dSupertrend10_3_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0