#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1w ADX(14) trend filter and 1d volume confirmation.
# Long when price breaks above Donchian upper band with weekly ADX > 25 (strong trend) and 1d volume > 1.8x 20-bar average.
# Short when price breaks below Donchian lower band with weekly ADX > 25 and 1d volume > 1.8x average.
# Exit when price closes below/above Donchian middle band.
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Weekly ADX ensures we only trade during strong trending regimes, avoiding chop/range whipsaws.
# 1d volume confirmation validates breakout significance using higher timeframe volume context.
# Donchian channels provide clear breakout levels effective in both trending and ranging markets.

name = "6h_Donchian20_1wADX14_1dVolumeConfirm"
timeframe = "6h"
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
    
    lookback = 20  # for Donchian and volume average
    
    # Calculate Donchian channels
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on 1w data
    if len(close_1w) < 14:
        adx_14_1w = np.full(len(close_1w), np.nan)
    else:
        # True Range
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_1w[1:] - high_1w[:-1]
        down_move = low_1w[:-1] - low_1w[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
        period = 14
        alpha = 1.0 / period
        tr_smooth = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(tr)
        minus_dm_smooth = np.zeros_like(tr)
        
        # Initialize with first period sum
        tr_smooth[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        # Wilder's smoothing for rest
        for i in range(period+1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.zeros_like(tr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # ADX = EMA of DX
        adx_14_1w = np.zeros_like(tr)
        adx_14_1w[2*period-1] = np.nanmean(dx[period:2*period])  # seed
        for i in range(2*period, len(dx)):
            adx_14_1w[i] = (adx_14_1w[i-1] * (period-1) + dx[i]) / period
        # Set NaN for insufficient data
        adx_14_1w[:2*period-1] = np.nan
    
    # Align 1w ADX to 6h timeframe (wait for 1w bar to close)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate average volume for confirmation (20-period) on 1d
    if len(volume_1d) < lookback:
        avg_volume_1d = np.full(len(volume_1d), np.nan)
    else:
        avg_volume_1d = pd.Series(volume_1d).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Align 1d average volume to 6h timeframe (wait for 1d bar to close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx_14_1w_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with strong weekly trend and 1d volume spike
            if (close[i] > donchian_upper[i] and 
                adx_14_1w_aligned[i] > 25 and 
                volume[i] > 1.8 * avg_volume_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower with strong weekly trend and 1d volume spike
            elif (close[i] < donchian_lower[i] and 
                  adx_14_1w_aligned[i] > 25 and 
                  volume[i] > 1.8 * avg_volume_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian middle (mean reversion)
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian middle (mean reversion)
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals