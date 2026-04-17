#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout (20-day) with weekly volume confirmation (vs 4-week average) and weekly ADX(14) trend filter
# Weekly ADX > 25 ensures we trade only in trending markets (avoids chop/range)
# Weekly volume spike > 1.5x 4-week average confirms institutional participation
# Designed for 1d timeframe to achieve 10-30 trades/year with low fee decay in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channel (20-week) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: 20-period high
    donch_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donch_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donch_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20_1w)
    donch_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20_1w)
    
    # === Weekly Volume Spike (vs 4-week average) ===
    volume_1w = df_1w['volume'].values
    vol_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4_1w)
    
    # === Weekly ADX (14-period) for trend filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])
    down_move = np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w
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
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: need 20 weeks for Donchian + 14 weeks for ADX + 4 weeks for volume MA
    # Each week = 7 days, so convert to days: (20+14)*7 = 238 days, plus buffer
    warmup = 250
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_20_1w_aligned[i]) or np.isnan(donch_low_20_1w_aligned[i]) or
            np.isnan(vol_ma_4_1w_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily price and weekly volume
        close_1d_aligned = align_htf_to_ltf(prices, df_1w, close)  # daily close aligned to weekly
        
        # Volume spike: current weekly volume > 1.5x 4-week average
        vol_spike = volume_1w[-1] > vol_ma_4_1w[-1] * 1.5 if len(volume_1w) > 0 else False
        # Proper alignment for current bar
        vol_spike_aligned = align_htf_to_ltf(prices, df_1w, 
            np.where(pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values > 0,
                    pd.Series(volume_1w).values > pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values * 1.5, False))
        vol_spike = vol_spike_aligned[i] if not np.isnan(vol_spike_aligned[i]) else False
        
        # Trend filter: weekly ADX > 25
        trend_filter = adx_1w_aligned[i] > 25
        
        # Donchian breakout signals (using weekly levels on daily price)
        breakout_up = close_1d_aligned[i] > donch_high_20_1w_aligned[i]
        breakout_down = close_1d_aligned[i] < donch_low_20_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and trend_filter:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when price returns to middle of weekly channel or conditions fail
        elif position == 1:
            # Exit long if price returns to midpoint or conditions fail
            midpoint = (donch_high_20_1w_aligned[i] + donch_low_20_1w_aligned[i]) / 2
            if close_1d_aligned[i] < midpoint or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to midpoint or conditions fail
            midpoint = (donch_high_20_1w_aligned[i] + donch_low_20_1w_aligned[i]) / 2
            if close_1d_aligned[i] > midpoint or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wVolume_ADXFilter"
timeframe = "1d"
leverage = 1.0