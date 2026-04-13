#!/usr/bin/env python3
"""
1h_4h_1d_Range_Breakout_With_Volume_Confirmation_and_Range_Filter
Hypothesis: In ranging markets (identified by low ADX and high Bollinger Band width percentile), 
breakouts from the previous 4-hour range with volume confirmation provide high-probability entries.
Use 1d ADX for regime filter (ADX < 25 = range), 4h range for breakout levels, and 1h for entry timing.
Volume > 1.5x 20-period average confirms institutional participation. Works in both bull and bear markets
by capturing volatility expansions after consolidation periods. Target: 15-30 trades/year per symbol.
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
    
    # Calculate ADX for regime filtering (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        adx_1d = np.full(len(prices), 100.0)  # Default to trending to avoid false signals
    else:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        
        # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            return result
        
        atr_1d = wilder_smooth(tr_1d, 14)
        dm_plus_smooth = wilder_smooth(dm_plus, 14)
        dm_minus_smooth = wilder_smooth(dm_minus, 14)
        
        # Avoid division by zero
        dx = np.where(atr_1d != 0, 
                      np.abs(dm_plus_smooth - dm_minus_smooth) / (dm_plus_smooth + dm_minus_smooth) * 100, 
                      0)
        adx_1d_raw = wilder_smooth(dx, 14)
        adx_1d = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    # Bollinger Band Width for range identification (1d timeframe)
    if len(df_1d) < 20:
        bb_width_pct_1d = np.full(len(prices), 50.0)  # Default to middle
    else:
        close_1d = df_1d['close'].values
        bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
        bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_width = bb_upper - bb_lower
        bb_width_pct = bb_width / bb_middle * 100
        # Percentile of current BB width over 50-period lookback
        bb_width_pct_1d_raw = pd.Series(bb_width_pct).rolling(window=50, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False)
        bb_width_pct_1d = align_htf_to_ltf(prices, df_1d, bb_width_pct_1d_raw.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # 4h range for breakout levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        range_high_4h = np.full(len(prices), np.nan)
        range_low_4h = np.full(len(prices), np.nan)
    else:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        # Previous 4h bar's high/low
        prev_high_4h = np.roll(high_4h, 1)
        prev_low_4h = np.roll(low_4h, 1)
        prev_high_4h[0] = np.nan
        prev_low_4h[0] = np.nan
        range_high_4h = align_htf_to_ltf(prices, df_4h, prev_high_4h)
        range_low_4h = align_htf_to_ltf(prices, df_4h, prev_low_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(adx_1d[i]) or np.isnan(bb_width_pct_1d[i]) or 
            np.isnan(range_high_4h[i]) or np.isnan(range_low_4h[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: ADX < 25 (ranging) AND BB width percentile > 50 (expanded range)
        range_condition = (adx_1d[i] < 25) and (bb_width_pct_1d[i] > 50)
        
        if not range_condition:
            signals[i] = 0.0
            continue
        
        # Long signal: break above 4h range with volume expansion
        long_signal = (close[i] > range_high_4h[i]) and volume_expansion[i]
        
        # Short signal: break below 4h range with volume expansion
        short_signal = (close[i] < range_low_4h[i]) and volume_expansion[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1d_Range_Breakout_With_Volume_Confirmation_and_Range_Filter"
timeframe = "1h"
leverage = 1.0