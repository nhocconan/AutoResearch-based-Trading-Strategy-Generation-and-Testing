#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and 1w ADX trend filter.
# Long when price breaks above R3 AND 1d volume > 2x 20-period average AND 1w ADX > 25.
# Short when price breaks below S3 AND 1d volume > 2x 20-period average AND 1w ADX > 25.
# Exit when price crosses back to R4/S4 (Camarilla danger zone).
# Uses Camarilla pivot structure with volume confirmation and weekly trend filter.
# Target: 100-180 total trades over 4 years (25-45/year) for low fee drift.

name = "4h_Camarilla_R3S3_1dVol_1wADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Camarilla levels (based on previous day)
    # Calculate daily pivot from previous day's OHLC
    # We'll use daily OHLC to calculate Camarilla for current 4h bar
    # For simplicity, we use previous day's close as proxy for pivot calculation
    # In practice, Camarilla uses (H+L+C)/3 from previous day
    # We'll approximate using rolling window
    
    # Calculate daily OHLC from 4h data (6 bars per day)
    # But to avoid look-ahead, we use previous completed day
    # We'll calculate daily values using 6-period rolling on 4h data
    
    # Previous day's high, low, close (using 6-period lag to avoid look-ahead)
    prev_day_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(6).values
    prev_day_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(6).values
    prev_day_close = pd.Series(close).rolling(window=6, min_periods=6).last().shift(6).values
    
    # Pivot point
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3
    range_val = prev_day_high - prev_day_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # 4h volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[np.isnan(dx)] = 0
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3, volume spike, ADX > 25
            long_cond = (close[i] > r3[i]) and volume_filter[i] and (adx_aligned[i] > 25)
            # Short conditions: break below S3, volume spike, ADX > 25
            short_cond = (close[i] < s3[i]) and volume_filter[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross back to R4 (danger zone) or below S3
            if close[i] >= r4[i] or close[i] <= s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross back to S4 (danger zone) or above R3
            if close[i] <= s4[i] or close[i] >= r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals