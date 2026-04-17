#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout (20-period) with 1-day volume confirmation (1.5x 20-period avg)
# and ADX trend filter (ADX > 25). Breakouts capture momentum; volume ensures conviction; ADX filters chop.
# Designed for 4h to achieve 20-50 trades/year with low fee decay. Works in bull (breakouts) and bear
# (ADX filters false breakouts in ranging markets).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data once
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian Channel (20-period) on 4h
    donch_high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20_4h)
    donch_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20_4h)
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day Volume Spike (vs 20-period average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1-day ADX (14-period) for trend filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_20_4h_aligned[i]) or np.isnan(donch_low_20_4h_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = volume_1d[i] > vol_ma_20_1d[i] * 1.5  # Use raw 1d values (already aligned via index)
        
        # Trend filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close_4h[i] > donch_high_20_4h_aligned[i]
        breakout_down = close_4h[i] < donch_low_20_4h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and trend_filter:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
        
        # Exit logic: exit when price returns to middle of channel or conditions fail
        elif position == 1:
            midpoint = (donch_high_20_4h_aligned[i] + donch_low_20_4h_aligned[i]) / 2
            if close_4h[i] < midpoint or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            midpoint = (donch_high_20_4h_aligned[i] + donch_low_20_4h_aligned[i]) / 2
            if close_4h[i] > midpoint or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolume_ADXFilter"
timeframe = "4h"
leverage = 1.0