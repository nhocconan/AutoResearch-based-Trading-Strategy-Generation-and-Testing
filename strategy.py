#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian channel breakouts on 6h timeframe filtered by weekly pivot
direction (from 1w HTF) and volume confirmation. Weekly pivot provides
longer-term structural bias: price above weekly pivot favors longs, below favors shorts.
This avoids counter-trend breakouts that fail in ranging/bear markets. Volume spike
confirms institutional participation. Designed for 12-37 trades/year to minimize fee drag
while capturing momentum in both bull (breakout continuation) and bear (mean reversion
from extremes when aligned with weekly trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    highest_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Get 1w data for weekly pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        weekly_pivot = np.full(n, np.nan)  # Will be handled in loop
    else:
        # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        weekly_pivot_raw = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot = align_htf_to_ltf(prices, df_1w, weekly_pivot_raw)
    
    # Calculate ATR(14) for stoploss on 6h data
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, ATR, and volume MA to propagate
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        weekly_piv = weekly_pivot[i] if len(df_1w) >= 1 else np.nan
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Weekly pivot filter: only trade in direction of weekly pivot bias
        # If no weekly data, default to neutral (allow both directions)
        if np.isnan(weekly_piv):
            weekly_long_bias = True   # Allow longs
            weekly_short_bias = True  # Allow shorts
        else:
            weekly_long_bias = curr_close > weekly_piv
            weekly_short_bias = curr_close < weekly_piv
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly pivot bias long AND volume spike
            long_condition = (curr_close > donchian_high) and weekly_long_bias and volume_spike
            # Short: price breaks below Donchian low AND weekly pivot bias short AND volume spike
            short_condition = (curr_close < donchian_low) and weekly_short_bias and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price breaks below Donchian low (reversal)
            if curr_close <= entry_price - 2.5 * atr_val or curr_close < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price breaks above Donchian high (reversal)
            if curr_close >= entry_price + 2.5 * atr_val or curr_close > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0