#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 12h Donchian breakout with daily volume confirmation and daily ADX trend filter.
# Long when price breaks above 12h Donchian upper channel AND daily volume > 1.3x 20-day average AND daily ADX > 20.
# Short when price breaks below 12h Donchian lower channel AND daily volume > 1.3x 20-day average AND daily ADX > 20.
# Exit when price crosses the 12h Donchian midpoint OR daily ADX drops below 15.
# Donchian provides clear breakout structure, daily volume confirms participation, daily ADX ensures trending conditions.
# Target: 12-37 trades/year per symbol (48-148 total over 4 years) to minimize fee drag while capturing major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    donchian_upper = np.full(len(high_12h), np.nan)
    donchian_lower = np.full(len(low_12h), np.nan)
    donchian_mid = np.full(len(close_12h), np.nan)
    
    for i in range(lookback - 1, len(high_12h)):
        donchian_upper[i] = np.max(high_12h[i - lookback + 1:i + 1])
        donchian_lower[i] = np.min(low_12h[i - lookback + 1:i + 1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Load daily data ONCE for ADX and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
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
    
    # Calculate 20-day average volume
    vol_ma_20 = np.full_like(volume_daily, np.nan)
    for i in range(19, len(volume_daily)):
        vol_ma_20[i] = np.mean(volume_daily[i-19:i+1])
    
    # Align indicators
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    volume_daily_aligned = align_htf_to_ltf(prices, df_daily, volume_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need daily and 12h data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        volume_ratio = volume_daily_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for breakout entries with volume confirmation and trend
            # Long: price breaks above upper channel AND volume > 1.3x average AND ADX > 20
            if (close[i] > donchian_upper_aligned[i] and 
                volume_ratio > 1.3 and 
                adx_aligned[i] > 20):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel AND volume > 1.3x average AND ADX > 20
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_ratio > 1.3 and 
                  adx_aligned[i] > 20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses midline or trend weakens
            if (close[i] < donchian_mid_aligned[i] or 
                adx_aligned[i] < 15):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses midline or trend weakens
            if (close[i] > donchian_mid_aligned[i] or 
                adx_aligned[i] < 15):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_12h_Donchian_Breakout_DailyVolume_ADX_v1"
timeframe = "12h"
leverage = 1.0