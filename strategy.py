#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with volume confirmation and 1w ADX trend filter.
Long when price > Alligator Jaw AND volume > 1.5x average AND 1w ADX > 25 (trending).
Short when price < Alligator Jaw AND volume > 1.5x average AND 1w ADX > 25.
Exit when price crosses Alligator Teeth OR 1w ADX < 20 (range market).
Uses 1d for Alligator calculation and 1w for ADX filter to reduce whipsaw.
Target: 30-100 total trades over 4 years (7-25/year). Alligator identifies trends,
volume confirmation filters fakeouts, weekly ADX filter avoids ranging markets.
Works in bull markets (captures uptrends) and bear markets (captures downtrends).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Alligator (Williams) on 1d timeframe
    # Jaw: 13-period SMMA smoothed by 8
    # Teeth: 8-period SMMA smoothed by 5
    # Lips: 5-period SMMA smoothed by 3
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # First value is SMA, then recursive smoothing
        smma_vals = np.full_like(series, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(series)):
            if not np.isnan(sma[i]) and not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + sma[i]) / period
            else:
                smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(close_1d, 13)  # Jaw (Blue)
    teeth = smma(close_1d, 8)  # Teeth (Red)
    lips = smma(close_1d, 5)   # Lips (Green)
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
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
    
    # Align 1d Alligator to 1d timeframe (no alignment needed)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period) on 1d
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Jaw AND volume > 1.5x avg AND ADX > 25 (trending)
            if price > jaw_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND volume > 1.5x avg AND ADX > 25 (trending)
            elif price < jaw_val and vol > 1.5 * vol_ma and adx_val > 25:
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

name = "1d_Alligator_Volume_1wADX_Filter"
timeframe = "1d"
leverage = 1.0