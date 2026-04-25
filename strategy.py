#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend direction and volume spike filter.
- Donchian Channel: 20-period high/low on 4h for breakout logic.
- Trend Filter: 1d EMA50 must align with breakout direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 4h volume > 1.8 * 20-period average 4h volume to confirm strong momentum.
- Entry: Long when close > Donchian High AND close > 1d EMA50 AND volume spike.
         Short when close < Donchian Low AND close < 1d EMA50 AND volume spike.
- Exit: Opposite Donchian break (long exits when close < Donchian Low, short exits when close > Donchian High).
- Signal size: 0.25 discrete to minimize fee drag.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_len = 20
    donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_len, 50, 20)  # Need 20 for Donchian, 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        donchian_high_level = donchian_high[i]
        donchian_low_level = donchian_low[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_dchigh = curr_close > donchian_high_level
        broke_below_dclow = curr_close < donchian_low_level
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below Donchian Low
            if position == 1:
                if curr_close < donchian_low_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above Donchian High
            elif position == -1:
                if curr_close > donchian_high_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above Donchian High AND above EMA50 AND volume spike
            long_condition = broke_above_dchigh and above_ema and volume_spike
            
            # Short: break below Donchian Low AND below EMA50 AND volume spike
            short_condition = broke_below_dclow and below_ema and volume_spike
            
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

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0