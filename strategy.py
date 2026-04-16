#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) with 1d ADX trend filter and volume confirmation.
# Long when price breaks above R4 with volume > 1.5x 20-period average and ADX > 25 (strong trend).
# Short when price breaks below S4 with volume > 1.5x 20-period average and ADX > 25.
# Fade longs at R3 when price rejects with volume > 1.5x average and ADX < 20 (range market).
# Fade shorts at S3 when price rejects with volume > 1.5x average and ADX < 20.
# Exit on opposite Camarilla level (R3/S3 for breakout, R4/S4 for fade) or volume drop below average.
# Uses discrete position size 0.25. Camarilla provides institutional pivot points, ADX filters regime,
# volume confirms institutional participation. Target: 75-150 total trades over 4 years (19-38/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly timeframe
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    R4_1w = pivot_1w + (range_1w * 1.1 / 2)
    R3_1w = pivot_1w + (range_1w * 1.1 / 4)
    S3_1w = pivot_1w - (range_1w * 1.1 / 4)
    S4_1w = pivot_1w - (range_1w * 1.1 / 2)
    
    # Align 1w Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4_1w)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4_1w)
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
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
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 6h data for volume moving average
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        r4 = R4_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        s4 = S4_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit breakout long if price drops below R3
            # Exit fade long if price rises above S3
            if (entry_price >= r4 and price < r3) or (entry_price <= r3 and price > s3):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit breakout short if price rises above S3
            # Exit fade short if price drops below R3
            if (entry_price <= s4 and price > s3) or (entry_price >= s3 and price < r3):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Breakout conditions: strong trend (ADX > 25)
            if adx_val > 25:
                # Breakout long: price > R4 with volume confirmation
                if price > r4 and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Breakout short: price < S4 with volume confirmation
                elif price < s4 and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            
            # Fade conditions: ranging market (ADX < 20)
            elif adx_val < 20:
                # Fade long: price < S3 with volume confirmation (expect bounce to R3)
                if price < s3 and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Fade short: price > R3 with volume confirmation (expect drop to S3)
                elif price > r3 and vol_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "6h_1wCamarillaR3S3R4S4_1dADX_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0