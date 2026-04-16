#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter.
# Long when price breaks above Donchian upper band AND 12h volume > 1.5x 20-period average AND 1d ADX > 25.
# Short when price breaks below Donchian lower band AND 12h volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price crosses the Donchian midpoint (mean of upper and lower bands).
# Uses discrete position size 0.25. Donchian channels provide structure, volume confirmation reduces false breakouts,
# 1d ADX ensures we only trade in strong trending regimes (works in both bull and bear markets).
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag while capturing major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(34) for trend filter ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
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
    tr_34 = pd.Series(tr).ewm(span=34, adjust=False, min_periods=34).mean().values
    dm_plus_34 = pd.Series(dm_plus).ewm(span=34, adjust=False, min_periods=34).mean().values
    dm_minus_34 = pd.Series(dm_minus).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_34 / tr_34
    di_minus = 100 * dm_minus_34 / tr_34
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Indicators: Donchian(20) channels ===
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_ub = high_12h  # upper band
    donchian_lb = low_12h   # lower band
    donchian_mid = (donchian_ub + donchian_lb) / 2.0  # midpoint for exit
    
    # Volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_ub[i]) or np.isnan(donchian_lb[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ub = donchian_ub[i]
        lb = donchian_lb[i]
        mid = donchian_mid[i]
        vol_ma_val = vol_ma_20[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1d ADX > 25 (strong trending regime)
        trend_filter = adx_val > 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian midpoint
            if price < mid:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian midpoint
            if price > mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian upper band with volume and trend confirmation
            if price > ub and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Donchian lower band with volume and trend confirmation
            elif price < lb and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_Donchian20_12hVolumeSpike_1dADX34Trend_V1"
timeframe = "12h"
leverage = 1.0