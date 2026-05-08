#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above Donchian(20) upper band AND volume > 1.5x 20-period average AND ADX > 25 (trending market).
# Short when price breaks below Donchian(20) lower band AND volume > 1.5x 20-period average AND ADX > 25.
# Exit when price crosses back inside the Donchian channel.
# Uses 12h timeframe as specified, with 1d volume and ADX for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "12h_Donchian_20_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Donchian(20) on 12h data
    donch_period = 20
    upper_band = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_band = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # ADX(14) on daily data
    adx_period = 14
    tr1 = np.maximum(df_d['high'] - df_d['low'], np.abs(df_d['high'] - np.roll(df_d['close'], 1)))
    tr2 = np.maximum(np.abs(df_d['low'] - np.roll(df_d['close'], 1)), tr1)
    tr = np.where(np.arange(len(df_d)) == 0, df_d['high'] - df_d['low'], tr2)
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # +DM and -DM
    up_move = df_d['high'] - np.roll(df_d['high'], 1)
    down_move = np.roll(df_d['low'], 1) - df_d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values
    atr_smooth = pd.Series(atr).rolling(window=adx_period, min_periods=adx_period).sum().values
    
    # Avoid division by zero
    plus_di = np.where(atr_smooth != 0, 100 * plus_dm_smooth / atr_smooth, 0)
    minus_di = np.where(atr_smooth != 0, 100 * minus_dm_smooth / atr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align daily volume filter and ADX to 12h timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, df_d, volume_filter)
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, 20, adx_period)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band, volume filter, ADX > 25
            long_cond = (close[i] > upper_band[i]) and volume_filter_aligned[i] and (adx_aligned[i] > 25)
            # Short conditions: price breaks below Donchian lower band, volume filter, ADX > 25
            short_cond = (close[i] < lower_band[i]) and volume_filter_aligned[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower band
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper band
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals