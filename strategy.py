#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and 6h volume confirmation.
# Long when price breaks above R3 (Camarilla resistance) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Short when price breaks below S3 (Camarilla support) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period average.
# Exit on break of opposite Camarilla level (R3 for longs, S3 for shorts) or ADX < 20 (range).
# Uses 1d HTF for ADX to avoid whipsaw in ranging markets. Volume confirmation reduces false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Camarilla levels provide precise support/resistance; ADX filter ensures trading only in trending regimes.

name = "6h_Camarilla_R3S3_Breakout_1dADX_6hVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Camarilla levels (based on previous day)
    # R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low), etc.
    # But we need previous day's levels, so shift by 1
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # first value invalid
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    diff = prev_high_1d - prev_low_1d
    R3 = prev_close_1d + 1.0 * diff
    S3 = prev_close_1d - 1.0 * diff
    
    # 1d ADX(14) - trend filter
    # Calculate True Range
    tr1 = prev_high_1d - prev_low_1d
    tr2 = np.abs(prev_high_1d - prev_close_1d)
    tr3 = np.abs(prev_low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((prev_high_1d - np.roll(prev_high_1d, 1)) > (np.roll(prev_low_1d, 1) - prev_low_1d),
                       np.maximum(prev_high_1d - np.roll(prev_high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(prev_low_1d, 1) - prev_low_1d) > (prev_high_1d - np.roll(prev_high_1d, 1)),
                        np.maximum(np.roll(prev_low_1d, 1) - prev_low_1d, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14, np.nan)
    di_minus = np.full_like(tr_14, np.nan)
    valid = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[valid] = (dm_plus_14[valid] / tr_14[valid]) * 100
    di_minus[valid] = (dm_minus_14[valid] / tr_14[valid]) * 100
    
    # DX and ADX
    dx = np.full_like(tr_14, np.nan)
    dx_valid = valid & ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])) * 100
    
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to LTF
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND ADX > 25 (trending) AND volume confirm
            if (close[i] > R3_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND ADX > 25 (trending) AND volume confirm
            elif (close[i] < S3_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 OR ADX < 20 (ranging)
            if (close[i] < S3_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R3 OR ADX < 20 (ranging)
            if (close[i] > R3_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals