#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (20-period average), 1w for EMA50 trend direction.
- Donchian channels: upper/lower bounds from 20-period high/low on 12h data.
- Entry: Long when price breaks above Donchian upper AND volume > 2.0 * 20-period average volume AND price > 1w EMA50.
         Short when price breaks below Donchian lower AND volume > 2.0 * 20-period average volume AND price < 1w EMA50.
- Exit: Opposite Donchian breakout (price < Donchian upper for long exit, price > Donchian lower for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Volume spike filters low-volume breakouts; 1w EMA50 ensures alignment with weekly trend.
- Works in bull markets (breakouts with volume) and bear markets (breakdowns with volume) with trend filter avoiding counter-trend trades.
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
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian upper: 20-period high, lower: 20-period low
    high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (already in 12h, but align to primary timeframe for consistency)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 50)  # Need 20 for Donchian/volume, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian upper
            if position == 1:
                if curr_close < donchian_upper_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian lower
            elif position == -1:
                if curr_close > donchian_lower_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian upper AND volume confirmation AND price > 1w EMA50
            long_condition = (curr_close > donchian_upper_aligned[i] and 
                            volume_confirm and
                            curr_close > ema50_1w_aligned[i])
            
            # Short: price breaks below Donchian lower AND volume confirmation AND price < 1w EMA50
            short_condition = (curr_close < donchian_lower_aligned[i] and 
                             volume_confirm and
                             curr_close < ema50_1w_aligned[i])
            
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

name = "12h_Donchian20_Breakout_1dVolumeSpike_1wEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0