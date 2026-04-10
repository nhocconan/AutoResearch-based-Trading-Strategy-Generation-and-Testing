#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX regime filter
# - Long when price breaks above 4h Donchian upper (20-period high) AND 1d volume > 1.5x 20-period average AND 1d ADX > 20 (trending)
# - Short when price breaks below 4h Donchian lower (20-period low) AND 1d volume > 1.5x 20-period average AND 1d ADX > 20
# - Exit when price returns to 4h Donchian midpoint (mean of upper/lower)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian channels provide objective breakout levels that work in both bull and bear markets
# - Volume confirmation reduces false breakouts during low-participation moves
# - Daily ADX filter ensures we trade only when higher timeframe is trending (avoids chop)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full_like(high, np.nan, dtype=float)
    lowest_low = np.full_like(low, np.nan, dtype=float)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
    def wheilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1d = wheilder_smoothing(tr, 14)
    dm_plus_smooth = wheilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wheilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1d, np.nan, dtype=float)
    di_minus = np.full_like(atr_1d, np.nan, dtype=float)
    for i in range(14, len(atr_1d)):
        if not np.isnan(atr_1d[i]) and atr_1d[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr_1d[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr_1d[i]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    for i in range(14, len(di_plus)):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = np.abs(di_plus[i] - di_minus[i]) / di_sum * 100
    
    adx_1d = wheilder_smoothing(dx, 14)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
            
            # Long conditions: price > upper band AND volume spike AND 1d trending (ADX > 20)
            if (close[i] > donchian_upper[i] and vol_spike and adx_1d_aligned[i] > 20):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < lower band AND volume spike AND 1d trending (ADX > 20)
            elif (close[i] < donchian_lower[i] and vol_spike and adx_1d_aligned[i] > 20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: price returns to midpoint
            exit_long = (position == 1 and close[i] < donchian_mid[i])
            exit_short = (position == -1 and close[i] > donchian_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def wheilder_smoothing(arr, period):
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value: simple average
    result[period-1] = np.nansum(arr[:period])
    # Subsequent values: Wilder's smoothing
    for i in range(period, len(arr)):
        if not np.isnan(result[i-1]):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
    return result