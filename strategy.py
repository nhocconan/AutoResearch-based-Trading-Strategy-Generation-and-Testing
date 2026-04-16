#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike (12h volume > 2.0x 50-period 12h volume median) and ADX > 20 regime filter
# Uses median-based volume threshold for robustness against outliers, and tighter ADX > 20 to capture more trending markets
# Long when price > upper Donchian(20) AND 12h volume > 2.0x 50-period 12h volume median AND ADX > 20
# Short when price < lower Donchian(20) AND 12h volume > 2.0x 50-period 12h volume median AND ADX > 20
# Exit on price returning to Donchian midpoint or ATR stoploss (1.5 ATR for tighter risk)
# Position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Get 1d data once before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Volume median (50-period) for robust confirmation ===
    volume_12h = df_12h['volume'].values
    vol_median_50_12h = pd.Series(volume_12h).rolling(window=50, min_periods=50).median().values
    vol_median_50_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_50_12h)
    
    # === 1d Indicator: ADX (14-period) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20, 28)  # 12h vol median, Donchian channels, 1d ADX need ~60 bars
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_median_50_12h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2.0x 50-period 12h volume median (more robust than mean)
        vol_threshold = vol_median_50_12h_aligned[i] * 2.0
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Regime filter: 1d ADX > 20 (trending market - lowered from 25 to capture more trends)
        trend_filter = adx_1d_aligned[i] > 20
        
        # Price levels
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        midpoint = donchian_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to midpoint or ATR stoploss
            if price <= midpoint or price <= entry_price - 1.5 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to midpoint or ATR stoploss
            if price >= midpoint or price >= entry_price + 1.5 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > upper Donchian AND volume confirmation AND trending market
            if price > upper and vol_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < lower Donchian AND volume confirmation AND trending market
            elif price < lower and vol_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_12hVolMedian2.0x_1dADX20_v1"
timeframe = "4h"
leverage = 1.0