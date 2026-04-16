#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above 12h Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Short when price breaks below 12h Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1w ADX > 25.
# Exit when price returns to 12h Camarilla pivot point (PP).
# Uses discrete position size 0.30. Volume confirmation reduces false signals, 1w ADX ensures strong trending regime.
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.
# Works in both bull (trend continuation) and bear (strong downtrends captured by shorts).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Camarilla pivot levels (using previous bar) ===
    # Camarilla formulas based on previous 12h bar
    pp = (high_12h + low_12h + close_12h) / 3.0
    r1 = pp + (high_12h - low_12h) * 1.1 / 12
    s1 = pp - (high_12h - low_12h) * 1.1 / 12
    r2 = pp + (high_12h - low_12h) * 1.1 / 6
    s2 = pp - (high_12h - low_12h) * 1.1 / 6
    r3 = pp + (high_12h - low_12h) * 1.1 / 4
    s3 = pp - (high_12h - low_12h) * 1.1 / 4
    r4 = pp + (high_12h - low_12h) * 1.1 / 2
    s4 = pp - (high_12h - low_12h) * 1.1 / 2
    
    # Shift by 1 to use previous bar levels (no look-ahead)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r4 = np.roll(r4, 1)
    s4 = np.roll(s4, 1)
    pp[0] = r1[0] = s1[0] = r2[0] = s2[0] = r3[0] = s3[0] = r4[0] = s4[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 1w data once before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: ADX(30) for strong trend filter ===
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
    tr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    dm_plus_30 = pd.Series(dm_plus).ewm(span=30, adjust=False, min_periods=30).mean().values
    dm_minus_30 = pd.Series(dm_minus).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_30 / tr_30
    di_minus = 100 * dm_minus_30 / tr_30
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(span=30, adjust=False, min_periods=30).mean().values
    
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using 1d volume MA)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1w ADX > 25 (strong trending regime)
        trend_filter = adx_val > 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Camarilla pivot point
            if price <= pp_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Camarilla pivot point
            if price >= pp_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Camarilla R3 with volume and trend confirmation
            if price > r3_val and vol_filter and trend_filter:
                signals[i] = 0.30
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S3 with volume and trend confirmation
            elif price < s3_val and vol_filter and trend_filter:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.30
    
    return signals

name = "12h_CamarillaR3S3_1dVolumeSpike_1wADXTrend_V1"
timeframe = "12h"
leverage = 1.0