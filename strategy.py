#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d volume spike and 1w ADX trend filter.
# Long when Alligator is bullish (Lips > Teeth > Jaw) AND price breaks above 4h Donchian upper(20) AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when Alligator is bearish (Lips < Teeth < Jaw) AND price breaks below 4h Donchian lower(20) AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Exit when Alligator changes direction or price returns to 4h Donchian midpoint.
# Uses discrete position size 0.25. Alligator provides trend direction, Donchian provides structure, volume confirmation reduces false signals,
# and 1w ADX ensures we only trade in trending regimes. Target: 50-120 total trades over 4 years (12-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian and Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: Donchian channels (20-period) ===
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    middle_aligned = align_htf_to_ltf(prices, df_4h, middle_20)
    
    # === 4h Indicators: Williams Alligator (13,8,5) ===
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(14) for trend filter ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1w timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        middle_val = middle_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using 1d volume MA)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1w ADX > 25 (trending regime)
        trend_filter = adx_val > 25
        
        # Alligator direction
        alligator_bullish = lips_val > teeth_val and teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Alligator turns bearish or price returns to Donchian middle
            if not alligator_bullish or price <= middle_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator turns bullish or price returns to Donchian middle
            if not alligator_bearish or price >= middle_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Alligator bullish AND price breaks above Donchian upper with volume and trend confirmation
            if alligator_bullish and price > upper_val and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Alligator bearish AND price breaks below Donchian lower with volume and trend confirmation
            elif alligator_bearish and price < lower_val and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Alligator_1dVolumeSpike_1wADXTrend_V1"
timeframe = "4h"
leverage = 1.0