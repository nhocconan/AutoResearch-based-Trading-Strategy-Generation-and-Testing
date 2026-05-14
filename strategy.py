#!/usr/bin/env python3
# Hypothesis: 6h Williams %R extreme readings combined with 1d ADX regime filter and 6h volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; ADX > 25 filters for trending markets to avoid false reversals in ranging conditions.
# Volume spike confirms institutional participation at extreme levels. Designed to work in both bull and bear markets by using regime filter.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams %R (14-period) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Extreme levels: oversold < -80, overbought > -20
    williams_r_oversold = williams_r < -80
    williams_r_overbought = williams_r > -20
    
    # 1d ADX (trend strength indicator) - HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    up_move = np.concatenate([[0], up_move])
    down_move = np.concatenate([[0], down_move])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    period = 14
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(up_move)
    minus_dm_sum = np.zeros_like(down_move)
    
    # Initial values
    tr_sum[period] = np.nansum(tr[1:period+1])
    plus_dm_sum[period] = np.nansum(up_move[1:period+1])
    minus_dm_sum[period] = np.nansum(down_move[1:period+1])
    
    # Wilder's smoothing
    for i in range(period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + up_move[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + down_move[i]
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.zeros_like(dx)
    
    # ADX is smoothed DX
    adx[2*period] = np.nanmean(dx[period+1:2*period+1])
    for i in range(2*period + 1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ADX > 25 indicates trending market (regime filter)
    adx_trending = adx_aligned > 25
    
    # Volume spike: > 1.8x 20-period average (moderate threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND trending market (ADX > 25) AND volume spike
            if (williams_r_oversold[i] and 
                adx_trending[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) AND trending market (ADX > 25) AND volume spike
            elif (williams_r_overbought[i] and 
                  adx_trending[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R rises above -50 (momentum weakening) OR ADX drops below 20 (trend ending)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R falls below -50 (momentum weakening) OR ADX drops below 20 (trend ending)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals