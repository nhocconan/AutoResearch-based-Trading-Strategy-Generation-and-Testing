#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Buy when price breaks above 20-day high + 1w ADX > 25 (trending) + volume > 1.5x 20-day average volume.
# Sell when price breaks below 20-day low + 1w ADX > 25 (trending) + volume spike.
# Uses 1d timeframe to reduce trade frequency (target: 10-25 trades/year) and minimize fee drag.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Works in both bull and bear markets by capturing strong trends in either direction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for price channel and volume (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on 1w
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/14)
    alpha = 1/14
    tr_14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    tr_14[13] = np.nansum(tr[14:28]) if not np.all(np.isnan(tr[14:28])) else np.nan
    dm_plus_14[13] = np.nansum(dm_plus[14:28]) if not np.all(np.isnan(dm_plus[14:28])) else 0
    dm_minus_14[13] = np.nansum(dm_minus[14:28]) if not np.all(np.isnan(dm_minus[14:28])) else 0
    for i in range(14, len(tr)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
        dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
        dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    # Avoid division by zero
    dm_plus_14 = np.where(tr_14 == 0, 0, dm_plus_14)
    dm_minus_14 = np.where(tr_14 == 0, 0, dm_minus_14)
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    # Smooth DX to get ADX
    adx_14 = np.full_like(dx, np.nan)
    if len(dx) >= 27:
        adx_14[27] = np.nanmean(dx[14:28]) if not np.all(np.isnan(dx[14:28])) else np.nan
        for i in range(28, len(dx)):
            adx_14[i] = adx_14[i-1] - (adx_14[i-1] / 14) + dx[i]
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume_1d[i]
        vol_ma = vol_ma_20[i]
        highest = highest_20[i]
        lowest = lowest_20[i]
        adx_val = adx_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above 20-day high + ADX > 25 + volume spike
            if price > highest and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-day low + ADX > 25 + volume spike
            elif price < lowest and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to middle of channel or trend weakens
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below 20-day low or ADX < 20
                if price < lowest or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above 20-day high or ADX < 20
                if price > highest or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wADX25_Volume"
timeframe = "1d"
leverage = 1.0