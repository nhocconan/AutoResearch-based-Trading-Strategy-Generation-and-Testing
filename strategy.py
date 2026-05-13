#!/usr/bin/env python3
# Hypothesis: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
# Williams %R(14) identifies overbought/oversold conditions on 6h.
# Long when %R < -80 (oversold) AND 1d ADX > 25 (trending market) AND volume > 2.0x 20-period average.
# Short when %R > -20 (overbought) AND 1d ADX > 25 AND volume > 2.0x average.
# Exit when %R crosses above -50 for long or below -50 for short.
# Uses 6h timeframe for lower frequency, Williams %R for mean reversion in trends, 1d ADX for regime filter, volume for confirmation.
# Target: 80-180 total trades over 4 years (20-45/year). Works in bull via buying dips in uptrend, bear via selling rallies in downtrend.

name = "6h_WilliamsR_1dADX_Trend_Volume_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Williams %R(14) on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = ((highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)) * -100
    # Replace division by zero or near-zero with -50 (neutral)
    denominator = highest_high_6h - lowest_low_6h
    williams_r_6h = np.where(denominator != 0, williams_r_6h, -50)
    
    # Volume filter: current 6h volume > 2.0x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (2.0 * vol_ma_6h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smooth = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    # Avoid division by zero
    dx = np.where((di_plus + di_minus) != 0,
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX is smoothed DX
    adx_1d = wilders_smoothing(dx, period_adx)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN or invalid
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i]) or adx_1d_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (strong trend) AND volume confirmation
            if williams_r_6h[i] < -80 and adx_1d_aligned[i] > 25 and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 AND volume confirmation
            elif williams_r_6h[i] > -20 and adx_1d_aligned[i] > 25 and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (exiting oversold)
            if williams_r_6h[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (exiting overbought)
            if williams_r_6h[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals