#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and ADX trend filter.
# Long when price breaks above weekly Donchian upper channel AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below weekly Donchian lower channel AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price crosses the weekly Donchian midpoint OR 1d ADX drops below 20.
# Weekly Donchian provides strong trend structure, 1d volume confirms participation, 1d ADX ensures trending conditions.
# Target: 10-30 trades/year per symbol (40-120 total over 4 years) to minimize fee drag while capturing major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(len(high_1w), np.nan)
    donchian_lower = np.full(len(low_1w), np.nan)
    donchian_mid = np.full(len(close_1w), np.nan)
    
    for i in range(lookback - 1, len(high_1w)):
        donchian_upper[i] = np.max(high_1w[i - lookback + 1:i + 1])
        donchian_lower[i] = np.min(low_1w[i - lookback + 1:i + 1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Load 1d data ONCE for ADX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average (skip first element for DM)
        result[period-1] = np.nanmean(data[1:period])  # Skip first element which is 0 for DM
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = np.full_like(dx, np.nan)
    # First ADX: simple average of first 14 DX values
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= 14:
        adx[13] = np.mean(valid_dx[:14])
        for i in range(14, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need 1d and 1w data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 1d volume vs 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_ratio = volume_1d_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for breakout entries with volume confirmation and trend
            # Long: price breaks above upper channel AND volume > 1.5x average AND ADX > 25
            if (close[i] > donchian_upper_aligned[i] and 
                volume_ratio > 1.5 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel AND volume > 1.5x average AND ADX > 25
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_ratio > 1.5 and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses midline or trend weakens
            if (close[i] < donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses midline or trend weakens
            if (close[i] > donchian_mid_aligned[i] or 
                adx_aligned[i] < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Breakout_Volume_ADX_v1"
timeframe = "1d"
leverage = 1.0