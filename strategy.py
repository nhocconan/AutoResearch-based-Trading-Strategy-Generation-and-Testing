#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_SqueezeBreakout
Hypothesis: In low volatility squeezes (Bollinger Band Width at 20-period low), price breaks out of Donchian(20) channel with volume confirmation (volume > 1.5x 20-period average volume). Works in both bull and breakout phases of bear markets by capturing explosive moves after consolidation. Uses 1d ADX to filter only when higher timeframe trend is strong (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = np.nanmean(data[max(0, i-period+1):i+1])
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    atr = wilders_smooth(tr, period)
    dm_plus_smooth = wilders_smooth(dm_plus, period)
    dm_minus_smooth = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Bollinger Bands on 4h
    bb_period = 20
    bb_std = 2.0
    
    # Simple moving average
    ma = np.full_like(close, np.nan)
    for i in range(bb_period-1, len(close)):
        ma[i] = np.mean(close[i-bb_period+1:i+1])
    
    # Standard deviation
    bb_std_dev = np.full_like(close, np.nan)
    for i in range(bb_period-1, len(close)):
        bb_std_dev[i] = np.std(close[i-bb_period+1:i+1])
    
    bb_upper = ma + bb_std * bb_std_dev
    bb_lower = ma - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (20-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 20
    for i in range(lookback-1, len(bb_width)):
        window = bb_width[i-lookback+1:i+1]
        if not np.all(np.isnan(window)):
            rank = np.sum(~np.isnan(window) & (bb_width[i] >= window))
            total = np.sum(~np.isnan(window))
            bb_width_percentile[i] = (rank / total) * 100 if total > 0 else 50
    
    # Squeeze condition: BB Width at 20-period low (bottom 10%)
    squeeze = bb_width_percentile <= 10
    
    # Donchian Channel (20-period)
    dc_period = 20
    dc_upper = np.full_like(high, np.nan)
    dc_lower = np.full_like(low, np.nan)
    
    for i in range(dc_period-1, len(high)):
        dc_upper[i] = np.max(high[i-dc_period+1:i+1])
        dc_lower[i] = np.min(low[i-dc_period+1:i+1])
    
    # Volume spike: volume > 1.5x 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    for i in range(vol_period-1, len(volume)):
        vol_ma[i] = np.mean(volume[i-vol_period+1:i+1])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Breakout conditions
    breakout_up = (close > dc_upper) & squeeze & volume_spike
    breakout_down = (close < dc_lower) & squeeze & volume_spike
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(50, dc_period, bb_period, vol_period, lookback), n):
        # Skip if ADX not available
        if np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only take trades when 1d ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            if breakout_up[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            elif breakout_down[i] and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below Donchian lower or ADX weakens
            if close[i] < dc_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above Donchian upper or ADX weakens
            if close[i] > dc_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_SqueezeBreakout"
timeframe = "4h"
leverage = 1.0