#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R reversal with 1-day ADX trend filter and volume confirmation.
Long when Williams %R crosses above -20 from below, 1-day ADX > 25, and volume spike.
Short when Williams %R crosses below -80 from above, 1-day ADX > 25, and volume spike.
Exit when Williams %R returns to -50 (mean reversion).
Williams %R captures overbought/oversold conditions; ADX filters for trending markets;
volume confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the daily trend.
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
    
    # Load 1-day data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = smooth_wilder(tr, 14)
    dm_plus14 = smooth_wilder(dm_plus, 14)
    dm_minus14 = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, 14)
    adx_1d = adx
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams %R (14-period) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after enough data for Williams %R
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R signals
        wr_cross_up = (williams_r[i] > -20) and (williams_r[i-1] <= -20)
        wr_cross_down = (williams_r[i] < -80) and (williams_r[i-1] >= -80)
        wr_mean_revert = (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
                         (williams_r[i] < -50 and williams_r[i-1] >= -50)
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below, ADX > 25, volume spike
            if wr_cross_up and adx_1d_aligned[i] > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above, ADX > 25, volume spike
            elif wr_cross_down and adx_1d_aligned[i] > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R mean reversion to -50
            if wr_mean_revert:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dADX_Trend_Volume"
timeframe = "4h"
leverage = 1.0