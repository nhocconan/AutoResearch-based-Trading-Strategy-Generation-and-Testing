#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above 12h Camarilla R3 AND 1d volume > 1.8x 20-period average AND 1w ADX > 25.
# Short when price breaks below 12h Camarilla S3 AND 1d volume > 1.8x 20-period average AND 1w ADX > 25.
# Exit when price returns to 12h Camarilla H5/L5 level.
# Uses discrete position size 0.25. Volume confirmation reduces false signals, 1w ADX ensures strong trending regime.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag.
# Works in bull/bear: ADX filter ensures we only trade strong trends, volume spike confirms institutional interest.

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
    
    # === 12h Indicators: Camarilla levels (based on previous day) ===
    # Camarilla levels calculated from previous 12h bar's OHLC
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # H5 = close + 1.1*(high-low)*1.1/2
    # L5 = close - 1.1*(high-low)*1.1/2
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # first bar
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    h5 = prev_close + 1.1 * camarilla_range * 1.1 / 2
    l5 = prev_close - 1.1 * camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    h5_aligned = align_htf_to_ltf(prices, df_12h, h5)
    l5_aligned = align_htf_to_ltf(prices, df_12h, l5)
    
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(h5_aligned[i]) or 
            np.isnan(l5_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        h5_val = h5_aligned[i]
        l5_val = l5_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period average (using 1d volume MA)
        vol_filter = vol > 1.8 * vol_ma_val if vol_ma_val > 0 else False
        
        # Trend filter: 1w ADX > 30 (strong trending regime)
        trend_filter = adx_val > 30
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Camarilla H5
            if price <= h5_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Camarilla L5
            if price >= l5_val:
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
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S3 with volume and trend confirmation
            elif price < s3_val and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR3S3_1dVolumeSpike_1wADXTrend_V1"
timeframe = "12h"
leverage = 1.0