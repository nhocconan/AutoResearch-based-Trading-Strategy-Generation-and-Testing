#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ADX trend filter.
# In trending markets (ADX > 25), breakout of 1d Donchian(20) signals continuation.
# Volume filter ensures momentum validity. Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in both bull (breakouts up) and bear (breakouts down) via symmetric long/short logic.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # === 1d Indicators: ADX (14) for trend filter ===
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_close_1d[0] = df_1d['close'].values[0]
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - prev_close_1d),
            np.abs(low_1d - prev_close_1d)
        )
    )
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move_1d = high_1d - np.roll(high_1d, 1)
    down_move_1d = np.roll(low_1d, 1) - low_1d
    up_move_1d[0] = 0
    down_move_1d[0] = 0
    plus_dm_1d = np.where((up_move_1d > down_move_1d) & (up_move_1d > 0), up_move_1d, 0)
    minus_dm_1d = np.where((down_move_1d > up_move_1d) & (down_move_1d > 0), down_move_1d, 0)
    
    # Smoothed +DM, -DM, TR
    tr_sum_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum_1d = pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum_1d = pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).sum().values
    
    # +DI and -DI
    plus_di_1d = 100 * (plus_dm_sum_1d / tr_sum_1d)
    minus_di_1d = 100 * (minus_dm_sum_1d / tr_sum_1d)
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === ENTRY LOGIC ===
        # Trending market: ADX > 25
        # Breakout long: price > 1d Donchian high
        # Breakout short: price < 1d Donchian low
        if adx_1d_aligned[i] > 25 and vol_confirm:
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.30  # long
            elif close[i] < donch_low_aligned[i]:
                signals[i] = -0.30  # short
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_1d_Donchian20_ADX25_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0