#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction to capture multi-week momentum.
- Donchian Channel: 20-period high/low from prior 1d candles for breakout logic.
- Trend Filter: 1w EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 1d volume > 1.5 * 20-period average 1d volume to confirm strong momentum.
- Entry: Long when close > upper Donchian AND close > 1w EMA50 AND volume spike.
         Short when close < lower Donchian AND close < 1w EMA50 AND volume spike.
- Exit: Opposite Donchian break (long exits when close < lower Donchian, short exits when close > upper Donchian).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with weekly trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation up) and bear markets (trend continuation down).
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian Channel (20-period) from prior day data
    # Use rolling window on daily data, then align
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Prior day's high/low for Donchian calculation (avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate 20-period rolling high/low on prior data
    high_roll_20 = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_roll_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_roll_20)
    
    # Calculate 1d volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for weekly EMA, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donchian[i]) or
            np.isnan(lower_donchian[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_level = upper_donchian[i]
        lower_level = lower_donchian[i]
        ema_50_level = ema_50_1w_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average volume
        volume_spike = curr_volume > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_level
        broke_below_lower = curr_close < lower_level
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower Donchian
            if position == 1:
                if curr_close < lower_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper Donchian
            elif position == -1:
                if curr_close > upper_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper Donchian AND above EMA50 AND volume spike
            long_condition = broke_above_upper and above_ema and volume_spike
            
            # Short: break below lower Donchian AND below EMA50 AND volume spike
            short_condition = broke_below_lower and below_ema and volume_spike
            
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0