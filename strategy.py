#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for trend strength (trending when ADX > 25, ranging when ADX < 20).
- Entry: Price breaks above/below 6h Camarilla H3/L3 levels with volume > 1.5 * 20-period volume MA and ADX > 25.
- Exit: Price touches opposite Camarilla level (L3 for long, H3 for short) or ADX drops below 20 (trend ends).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by only trading strong trends (ADX > 25) while using Camarilla levels for precise entries.
Volume confirmation reduces false breakouts, and ADX regime filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla levels and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 6h Camarilla levels (based on previous day's OHLC)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla levels use previous period's OHLC (6h timeframe)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # We need to shift by 1 to use previous period's values
    prev_close = np.roll(close_6h, 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close[0] = close_6h[0]  # first value
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range / 2
    l3 = prev_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_6h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_6h, l3)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Handle division by zero and NaN
    adx = np.nan_to_num(adx, nan=0.0)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 14, 20)  # Camarilla, ADX calculation, ADX smoothing, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and ADX > 25 (trending)
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            strong_trend = adx_aligned[i] > 25
            
            # Long: price breaks above H3 AND strong trend AND volume confirmed
            if curr_high > h3_aligned[i] and strong_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND strong trend AND volume confirmed
            elif curr_low < l3_aligned[i] and strong_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price touches L3 or ADX drops below 20 (trend ending)
            if curr_low <= l3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price touches H3 or ADX drops below 20 (trend ending)
            if curr_high >= h3_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0