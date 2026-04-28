#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike (>2x 20-bar avg) and ADX25 regime filter
# Long when price > Donchian high(20) AND volume > 2x 20-bar avg AND ADX > 25
# Short when price < Donchian low(20) AND volume > 2x 20-bar avg AND ADX > 25
# Exit when price crosses Donchian midpoint OR volume drops below 1.5x 20-bar avg
# Target: 12-37 trades/year via tight entry conditions combining breakout, volume, and regime filters
# Works in bull markets via breakouts and bear markets via breakdowns, with ADX filtering ranging periods

name = "12h_Donchian20_1dVolumeSpike_ADX25_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for volume MA and ADX
        return np.zeros(n)
    
    # Calculate 20-bar volume MA on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (since we lost first bar in calculations)
    volume_ma_20_1d = np.concatenate([np.full(19, np.nan), volume_ma_20_1d])  # 20-bar MA loses 19 bars
    adx = np.concatenate([np.full(27, np.nan), adx])  # 1 (TR) + 14 (TR smoothing) + 14 (ADX smoothing) - 2
    
    # Align 1d indicators to 12h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 12h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: >2x 20-bar average volume on 1d
    volume_spike = volume > 2.0 * volume_ma_20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, lookback)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price > Donchian high AND volume spike AND ADX > 25 (trending)
            if price > upper and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short when price < Donchian low AND volume spike AND ADX > 25 (trending)
            elif price < lower and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses midpoint OR volume drops below 1.5x
            if price < mid or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses midpoint OR volume drops below 1.5x
            if price > mid or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals