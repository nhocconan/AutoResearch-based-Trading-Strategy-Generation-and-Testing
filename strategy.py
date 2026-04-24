#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d ADX(14) > 25 for strong trend filtering (avoids choppy markets).
- Williams %R(14) on 4h: long when < -90 (oversold), short when > -10 (overbought).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Entry: Long when Williams %R < -90 AND 1d ADX > 25 AND volume spike.
         Short when Williams %R > -10 AND 1d ADX > 25 AND volume spike.
- Exit: Williams %R reverts to > -50 (for long) or < -50 (for short) OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe.
This strategy captures mean reversions in strong trends, avoiding counter-trend trades.
Williams %R identifies exhaustion points, ADX ensures we only trade in strong trends,
and volume confirmation filters for institutional participation. Works in both bull and bear
markets by only taking trades in the direction of the strong trend, with Williams %R
providing precise entry timing at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 4h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_4h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_4h['low']).rolling(window=14, min_periods=14).min().values
    close_4h = df_4h['close'].values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for ADX and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period 1d volume MA
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need enough bars for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_williams_r = williams_r_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend (ADX > 25)
            if volume_spike[i] and adx_aligned[i] > 25:
                # Bullish entry: Williams %R < -90 (oversold)
                if curr_williams_r < -90:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -10 (overbought)
                elif curr_williams_r > -10:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to > -50 OR loss of volume confirmation OR weak trend
            if curr_williams_r > -50 or not volume_spike[i] or adx_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to < -50 OR loss of volume confirmation OR weak trend
            if curr_williams_r < -50 or not volume_spike[i] or adx_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0