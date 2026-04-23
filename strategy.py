#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume spike confirmation.
Long when price breaks above Donchian upper AND 1d ADX > 25 (trending) AND 1h volume > 1.8x 20-period MA.
Short when price breaks below Donchian lower AND 1d ADX > 25 (trending) AND 1h volume > 1.8x 20-period MA.
Exit when price touches opposite Donchian level or 1d ADX < 20 (range regime).
Uses 1d HTF for regime filter to avoid whipsaws in ranging markets, volume spike for momentum confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian provides clear structure, ADX regime filter improves bear market performance.
"""

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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_middle = np.full(n, np.nan)
    
    for i in range(20, n):  # Need 20 periods for calculation
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
        donchian_middle[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30, 20)  # Donchian (20), ADX (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 1h volume > 1.8x 20-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_val
        
        # Regime filter: 1d ADX > 25 indicates trending market
        trending_regime = adx_val > 25
        ranging_regime = adx_val < 20  # Exit when ranging
        
        if position == 0:
            # Long: Break above Donchian upper AND trending regime AND volume filter
            if price > upper and trending_regime and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND trending regime AND volume filter
            elif price < lower and trending_regime and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower OR ADX drops below 20 (ranging)
                if price < lower or ranging_regime:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper OR ADX drops below 20 (ranging)
                if price > upper or ranging_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dADX_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0