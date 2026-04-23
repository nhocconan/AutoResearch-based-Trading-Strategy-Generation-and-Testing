#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1d ADX > 25 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 AND 1d ADX > 25 AND volume > 1.8x 20-period average.
Exit when price touches opposite Camarilla level (R2/S2) or ADX < 20 (trend weak).
Uses 1d HTF for ADX trend strength (avoids false breakouts in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 1d ADX for trend strength filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We need previous day's OHLC for each 6h bar
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    o_1d = df_1d_for_camarilla['open'].values
    h_1d = df_1d_for_camarilla['high'].values
    l_1d = df_1d_for_camarilla['low'].values
    c_1d = df_1d_for_camarilla['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3_1d = np.full_like(c_1d, np.nan)
    camarilla_s3_1d = np.full_like(c_1d, np.nan)
    camarilla_r2_1d = np.full_like(c_1d, np.nan)
    camarilla_s2_1d = np.full_like(c_1d, np.nan)
    
    for i in range(len(c_1d)):
        if np.isnan(h_1d[i]) or np.isnan(l_1d[i]) or np.isnan(c_1d[i]):
            continue
        rang = h_1d[i] - l_1d[i]
        camarilla_r3_1d[i] = c_1d[i] + (rang * 1.1 / 4)
        camarilla_s3_1d[i] = c_1d[i] - (rang * 1.1 / 4)
        camarilla_r2_1d[i] = c_1d[i] + (rang * 1.1 / 6)
        camarilla_s2_1d[i] = c_1d[i] - (rang * 1.1 / 6)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s3_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s2_1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND ADX > 25 AND volume spike
            if price > r3 and adx_val > 25 and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND ADX > 25 AND volume spike
            elif price < s3 and adx_val > 25 and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S2 OR ADX < 20 (trend weak)
                if price < s2 or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R2 OR ADX < 20 (trend weak)
                if price > r2 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dADX_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0