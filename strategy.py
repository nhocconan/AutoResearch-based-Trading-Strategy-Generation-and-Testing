#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with volume confirmation and 1d ADX trend filter.
Long when price > Alligator Jaw (13-period SMMA shifted 8) AND volume > 1.5x average AND ADX > 25 (trending).
Short when price < Alligator Lips (8-period SMMA shifted 5) AND volume > 1.5x average AND ADX > 25.
Exit when price crosses Alligator Teeth (5-period SMMA shifted 3) OR ADX < 20 (range market).
Uses 12h for Alligator calculation and 1d for ADX filter to reduce whipsaw.
Target: 50-150 total trades over 4 years (12-37/year). Alligator identifies trend direction,
volume confirmation filters fakeouts, ADX filter avoids ranging markets.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.empty(len(values))
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Value) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Alligator components on 12h timeframe
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_12h = (high_12h + low_12h) / 2
    jaw_raw = smma(median_12h, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (into future)
    jaw[:8] = np.nan  # first 8 values invalid after shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_12h, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_12h, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h Alligator to 12h timeframe
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 12h
    volume_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Jaw AND volume > 1.5x avg AND ADX > 25 (trending)
            if price > jaw_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Lips AND volume > 1.5x avg AND ADX > 25 (trending)
            elif price < lips_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Teeth OR ADX < 20 (range market)
            if price < teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Teeth OR ADX < 20 (range market)
            if price > teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Alligator_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0