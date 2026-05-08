#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ADX for trend strength, 4h Donchian(20) breakout for entry,
# and volume confirmation. Long when ADX > 25, price breaks above 4h Donchian upper band,
# volume > 1.5x average. Short when ADX > 25, price breaks below 4h Donchian lower band,
# volume > 1.5x average. Exit when price crosses opposite Donchian band or ADX < 20.
# Position size fixed at 0.25 to limit risk and reduce trade frequency.
# Target: 20-50 trades/year to avoid fee drag. Works in trending markets (ADX > 25).
# Avoids range-bound markets where breakouts fail.

name = "4h_ADX25_Donchian20_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate ADX(14) on 1d
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1e-10, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_above_25 = adx > 25
    adx_below_20 = adx < 20
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d ADX to 4h
    adx_above_25_aligned = align_htf_to_ltf(prices, df_1d, adx_above_25.astype(float))
    adx_below_20_aligned = align_htf_to_ltf(prices, df_1d, adx_below_20.astype(float))
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 40  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_above_25_aligned[i]) or np.isnan(adx_below_20_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25, price breaks above 4h Donchian upper band, volume spike
            if (adx_above_25_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: ADX > 25, price breaks below 4h Donchian lower band, volume spike
            elif (adx_above_25_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: ADX < 20 (trend weak) or price breaks below 4h Donchian lower band
            if (adx_below_20_aligned[i] or 
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (trend weak) or price breaks above 4h Donchian upper band
            if (adx_below_20_aligned[i] or 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals