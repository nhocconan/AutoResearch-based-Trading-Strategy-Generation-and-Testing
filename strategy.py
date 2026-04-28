#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike (2x 20-bar avg) and ADX(14) > 25 trend filter
# Long when price breaks above upper Donchian channel AND volume > 2x 20-bar avg AND ADX > 25
# Short when price breaks below lower Donchian channel AND volume > 2x 20-bar avg AND ADX > 25
# Exit when price returns to middle of Donchian channel (mean reversion) or volume drops below 1.5x
# Uses 1d HTF for volume and ADX to avoid lower timeframe noise
# Target: 12-37 trades/year via tight entry conditions requiring confluence of breakout, volume, and trend

name = "12h_Donchian20_1dVolumeSpike_ADX25_TrendFilter_v1"
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
    if len(df_1d) < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Calculate volume 20-bar MA on 1d
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
    volume_ma_20_1d = np.concatenate([np.full(19, np.nan), volume_ma_20_1d])  # 20 MA needs 19 padding
    adx = np.concatenate([np.full(27, np.nan), adx])  # 1 (TR) + 14 (TR smoothing) + 14 (ADX smoothing) - 2
    
    # Align 1d indicators to 12h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 12h data (20-bar lookback)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(volume_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_current = volume[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        price = close[i]
        
        # Volume confirmation: >2x 20-bar average volume (1d)
        volume_spike = vol_current > 2.0 * vol_ma
        # Trend filter: ADX > 25
        trending = adx_val > 25
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND volume spike AND trending
            if price > upper and volume_spike and trending:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND volume spike AND trending
            elif price < lower and volume_spike and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to middle or volume drops
            if price < middle or not (volume_spike and trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to middle or volume drops
            if price > middle or not (volume_spike and trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals