# This strategy implements a 4-hour timeframe approach using 12-hour timeframe indicators
# Strategy combines 12h Donchian breakout with volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian upper channel AND 4h volume > 1.2x 20-period average AND 12h ADX > 25
# Short when price breaks below 12h Donchian lower channel AND 4h volume > 1.2x 20-period average AND 12h ADX > 25
# Exit when price crosses the 12h Donchian midpoint OR 12h ADX drops below 20
# Position size: 0.25 (25% of capital) to manage risk during drawdowns
# Designed to work in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    donchian_upper_12h = np.full(len(high_12h), np.nan)
    donchian_lower_12h = np.full(len(low_12h), np.nan)
    donchian_mid_12h = np.full(len(close_12h), np.nan)
    
    for i in range(lookback - 1, len(high_12h)):
        donchian_upper_12h[i] = np.max(high_12h[i - lookback + 1:i + 1])
        donchian_lower_12h[i] = np.min(low_12h[i - lookback + 1:i + 1])
        donchian_mid_12h[i] = (donchian_upper_12h[i] + donchian_lower_12h[i]) / 2.0
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
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
    adx_12h = np.full_like(dx, np.nan)
    # First ADX: simple average of first 14 DX values
    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) >= 14:
        adx_12h[13] = np.mean(valid_dx[:14])
        for i in range(14, len(dx)):
            if not np.isnan(dx[i]):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    # Calculate 20-period average volume (using 4h data for volume confirmation)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need 12h data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period average
        volume_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        if position == 0:
            # Look for breakout entries with volume confirmation and trend
            # Long: price breaks above upper channel AND volume > 1.2x average AND ADX > 25
            if (close[i] > donchian_upper_aligned[i] and 
                volume_ratio > 1.2 and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel AND volume > 1.2x average AND ADX > 25
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_ratio > 1.2 and 
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

name = "4h_12h_Donchian_Breakout_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0