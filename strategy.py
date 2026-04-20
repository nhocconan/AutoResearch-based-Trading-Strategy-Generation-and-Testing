#!/usr/bin/env python3
# 4h_1D_Donchian_Breakout_VolumeTrend_Regime
# Hypothesis: On 4h timeframe, trade breakouts of daily Donchian channels with volume confirmation and trend regime filter.
# In trending markets (ADX > 25), trade breakouts in direction of trend. In ranging markets (ADX < 25), fade at channel extremes.
# Uses 1d ADX to filter regime and 20-period volume average for confirmation. Targets 20-40 trades/year.

name = "4h_1D_Donchian_Breakout_VolumeTrend_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands
    upper_1d = np.full_like(high_1d, np.nan)
    lower_1d = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Calculate 1d ADX for trend/ranging filter (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR and DM using Wilder's smoothing
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    
    # Align 1d indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (ADX > 25): breakout in direction of trend
            if adx_aligned[i] > 25:
                # Long breakout above upper band with volume confirmation
                if (close[i] > upper_aligned[i] * 1.002 and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Short breakdown below lower band with volume confirmation
                elif (close[i] < lower_aligned[i] * 0.998 and 
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
            # Ranging market (ADX < 25): fade at channel extremes
            elif adx_aligned[i] < 25:
                # Long near lower band with volume confirmation
                if (close[i] <= lower_aligned[i] * 1.002 and 
                    close[i] >= lower_aligned[i] * 0.998 and
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.30
                    position = 1
                # Short near upper band with volume confirmation
                elif (close[i] >= upper_aligned[i] * 0.998 and 
                      close[i] <= upper_aligned[i] * 1.002 and
                      volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: reverse at opposite band or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] >= upper_aligned[i] * 0.998) or \
               (adx_aligned[i] > 25 and close[i] < lower_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: reverse at opposite band or ADX shifts to ranging
            if (adx_aligned[i] < 25 and close[i] <= lower_aligned[i] * 1.002) or \
               (adx_aligned[i] > 25 and close[i] > upper_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals