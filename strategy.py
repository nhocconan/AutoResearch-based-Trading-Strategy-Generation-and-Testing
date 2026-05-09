#!/usr/bin/env python3
# 6h_Chandelier_Exit_Volume_Regime
# Hypothesis: Chandelier Exit (trailing stop based on ATR) combined with volume confirmation and ADX regime filter.
# Long when price crosses above Chandelier Exit long level in strong uptrend (ADX>25), short when crosses below short level in strong downtrend.
# Volume confirmation (>1.5x 20-period average) ensures breakout strength.
# Designed for low trade frequency (<30/year) to minimize fee drift in 6h timeframe.
# Works in both bull and bear markets by following trend direction with dynamic stop.

name = "6h_Chandelier_Exit_Volume_Regime"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for Chandelier Exit calculation (using 22-period ATR and 3x multiplier)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for 12h
    tr_12h = np.zeros_like(high_12h)
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    # Calculate ATR(22) for 12h
    atr_12h = np.full_like(high_12h, np.nan)
    if len(tr_12h) >= 22:
        atr_12h[21] = np.mean(tr_12h[0:22])
        for i in range(22, len(tr_12h)):
            atr_12h[i] = (atr_12h[i-1] * 21 + tr_12h[i]) / 22
    
    # Calculate Chandelier Exit levels for 12h
    # Long exit: highest high - 3*ATR
    # Short exit: lowest low + 3*ATR
    highest_high_12h = np.full_like(high_12h, np.nan)
    lowest_low_12h = np.full_like(low_12h, np.nan)
    
    if len(high_12h) >= 22:
        for i in range(22, len(high_12h)):
            highest_high_12h[i] = np.max(high_12h[i-21:i+1])
            lowest_low_12h[i] = np.min(low_12h[i-21:i+1])
    
    chandelier_long_exit_12h = np.full_like(high_12h, np.nan)
    chandelier_short_exit_12h = np.full_like(high_12h, np.nan)
    
    valid_atr = ~np.isnan(atr_12h)
    if np.any(valid_atr):
        chandelier_long_exit_12h[valid_atr] = highest_high_12h[valid_atr] - 3.0 * atr_12h[valid_atr]
        chandelier_short_exit_12h[valid_atr] = lowest_low_12h[valid_atr] + 3.0 * atr_12h[valid_atr]
    
    # Align Chandelier Exit levels to 6h timeframe
    chandelier_long_exit_aligned = align_htf_to_ltf(prices, df_12h, chandelier_long_exit_12h)
    chandelier_short_exit_aligned = align_htf_to_ltf(prices, df_12h, chandelier_short_exit_12h)
    
    # Get 1d data for ADX calculation (trend strength)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr_1d = np.zeros_like(high_1d)
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Calculate +DM and -DM for 1d
    plus_dm_1d = np.zeros_like(high_1d)
    minus_dm_1d = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Calculate smoothed TR, +DM, -DM (14-period)
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[0:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = smooth_series(tr_1d, 14)
    plus_dm_smoothed = smooth_series(plus_dm_1d, 14)
    minus_dm_smoothed = smooth_series(minus_dm_1d, 14)
    
    # Calculate +DI and -DI
    plus_di_1d = np.full_like(high_1d, np.nan)
    minus_di_1d = np.full_like(high_1d, np.nan)
    valid_atr_1d = ~np.isnan(atr_1d) & (atr_1d != 0)
    if np.any(valid_atr_1d):
        plus_di_1d[valid_atr_1d] = 100 * plus_dm_smoothed[valid_atr_1d] / atr_1d[valid_atr_1d]
        minus_di_1d[valid_atr_1d] = 100 * minus_dm_smoothed[valid_atr_1d] / atr_1d[valid_atr_1d]
    
    # Calculate DX and ADX
    dx_1d = np.full_like(high_1d, np.nan)
    di_sum = plus_di_1d + minus_di_1d
    valid_di = ~np.isnan(di_sum) & (di_sum != 0)
    if np.any(valid_di):
        dx_1d[valid_di] = 100 * np.abs(plus_di_1d[valid_di] - minus_di_1d[valid_di]) / di_sum[valid_di]
    
    adx_1d = smooth_series(dx_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(22, 20, 28)  # Ensure Chandelier (22), volume MA (20), and ADX (14+14) are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chandelier_long_exit_aligned[i]) or np.isnan(chandelier_short_exit_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above Chandelier long exit AND strong uptrend (ADX>25) AND volume confirmation
            if (close[i] > chandelier_long_exit_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below Chandelier short exit AND strong downtrend (ADX>25) AND volume confirmation
            elif (close[i] < chandelier_short_exit_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Chandelier long exit OR trend weakens (ADX<20)
            if close[i] < chandelier_long_exit_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Chandelier short exit OR trend weakens (ADX<20)
            if close[i] > chandelier_short_exit_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals