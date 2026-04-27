#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dADX_VolumeSpike_Regime
Hypothesis: 4h strategy using Donchian(20) breakouts with 1d ADX>25 trend filter and volume spike confirmation. 
Enter long when price closes above upper Donchian band with 1d uptrend (ADX>25 and +DI>-DI) and volume > 2.0x 20-period average. 
Enter short when price closes below lower Donchian band with 1d downtrend (ADX>25 and +DI<+DI) and volume confirmation. 
Exit on opposite Donchian band touch or 1d ADX<20 (range regime). 
Designed for moderate trade frequency (~30-60/year) with discrete position sizing (0.25) to balance edge and fee drag.
Works in both bull and bear markets by following the 1d trend regime while using Donchian breakouts for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for ADX
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # 1d ADX calculation (14-period)
    period = 14
    # True Range
    tr1 = np.abs(h_1d[1:] - l_1d[1:])
    tr2 = np.abs(h_1d[1:] - c_1d[:-1])
    tr3 = np.abs(l_1d[1:] - c_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0.0]), tr])  # first tr = 0
    
    # Directional Movement
    up_move = h_1d[1:] - h_1d[:-1]
    down_move = l_1d[:-1] - l_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([np.array([0.0]), plus_dm])
    minus_dm = np.concatenate([np.array([0.0]), minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_period = wilders_smoothing(tr, period)
    plus_dm_period = wilders_smoothing(plus_dm, period)
    minus_dm_period = wilders_smoothing(minus_dm, period)
    
    # DI and ADX
    plus_di = 100 * plus_dm_period / tr_period
    minus_di = 100 * minus_dm_period / tr_period
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # 1d trend conditions: ADX>25 and +DI>-DI for uptrend, ADX>25 and +DI<+DI for downtrend
    adx_gt_25 = adx > 25
    plus_di_gt_minus_di = plus_di > minus_di
    plus_di_lt_minus_di = plus_di < minus_di
    
    uptrend_1d = adx_gt_25 & plus_di_gt_minus_di
    downtrend_1d = adx_gt_25 & plus_di_lt_minus_di
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h Donchian channels (20-period)
    donchian_len = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), volume avg (20), ADX (14+14=28 min for stability)
    start_idx = max(donchian_len, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_val = upper_donchian[i]
        lower_val = lower_donchian[i]
        uptrend_val = uptrend_aligned[i] > 0.5
        downtrend_val = downtrend_aligned[i] > 0.5
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1d trend filter and volume spike
            # Long: price closes above upper Donchian AND 1d uptrend
            long_condition = (close_val > upper_val) and uptrend_val and vol_conf
            # Short: price closes below lower Donchian AND 1d downtrend
            short_condition = (close_val < lower_val) and downtrend_val and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian (opposite band) OR 1d trend weakens (ADX<20)
            if (close_val < lower_val) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches upper Donchian (opposite band) OR 1d trend weakens (ADX<20)
            if (close_val > upper_val) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dADX_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0