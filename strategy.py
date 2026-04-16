#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 4h volume confirmation and 12h ADX(20) trend filter.
# Long when price breaks above Donchian upper band AND 4h volume > 1.8x 20-period average AND 12h ADX > 20.
# Short when price breaks below Donchian lower band AND 4h volume > 1.8x 20-period average AND 12h ADX > 20.
# Exit when price returns to Donchian midpoint (mean of upper and lower bands).
# Uses discrete position size 0.30. Donchian provides objective trend-following structure.
# Volume confirmation reduces false breakouts, 12h ADX ensures higher timeframe trend alignment.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for ADX filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: ADX(20) for trend filter ===
    if len(high_12h) < 20:
        return np.zeros(n)
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (using Wilder's smoothing via EMA with alpha=1/period)
    tr_20 = pd.Series(tr).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    dm_plus_20 = pd.Series(dm_plus).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    dm_minus_20 = pd.Series(dm_minus).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_20 / tr_20
    di_minus = 100 * dm_minus_20 / tr_20
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 4h Indicators: Donchian(20) and Volume MA ===
    # Donchian upper/lower bands (20-period high/low)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # 4h Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        vol_ma_val = vol_ma_20[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period average
        vol_filter = vol > 1.8 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 12h ADX > 20 (trending regime)
        trend_filter = adx_val > 20
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian midpoint
            if price <= mid:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian midpoint
            if price >= mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper band with volume and trend confirmation
            if price > upper and vol_filter and trend_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with volume and trend confirmation
            elif price < lower and vol_filter and trend_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "4h_Donchian20_4hVolumeSpike_12hADXTrend_V1"
timeframe = "4h"
leverage = 1.0