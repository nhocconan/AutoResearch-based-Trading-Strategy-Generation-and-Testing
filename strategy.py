#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakout traded only in the direction of the weekly pivot trend (bullish if weekly close > weekly open, bearish otherwise). Uses volume confirmation (2.0x average) to reduce false breakouts. Designed for low trade frequency (target 12-37/year) to overcome fee drag in ranging/bear markets. Discrete position sizing (0.25) minimizes churn. Works in bull markets via breakouts with weekly trend and in bear via fade at extremes with volume exhaustion filtering. Focus on BTC/ETH as primary targets.
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
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot direction: 1 if bullish (close > open), -1 if bearish (close < open), 0 if doji
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_direction = np.where(weekly_close > weekly_open, 1, np.where(weekly_close < weekly_open, -1, 0))
    weekly_direction_aligned = align_htf_to_ltf(prices, df_1w, weekly_direction)
    
    # Get 1d data for volume average (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average (using 1d volume for stability)
    # We need to align 1d volume average to 6h
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), ATR(14), volume MA(20)
    start_idx = max(20, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_direction_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        weekly_dir = weekly_direction_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly trend bullish AND volume spike
            long_signal = (close_val > highest_high[i]) and (weekly_dir == 1) and vol_spike
            
            # Short: price breaks below Donchian low AND weekly trend bearish AND volume spike
            short_signal = (close_val < lowest_low[i]) and (weekly_dir == -1) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly trend flips bearish OR price hits ATR stoploss
            if (weekly_dir == -1) or (close_val < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend flips bullish OR price hits ATR stoploss
            if (weekly_dir == 1) or (close_val > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0