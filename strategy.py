#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R3S3_Reversion
# Hypothesis: Mean reversion at weekly Camarilla R3/S3 levels with daily volume confirmation and ADX filter.
# Trades only when price touches weekly R3/S3 with high volume and low ADX (range market).
# Works in bull/bear by fading extremes in ranging conditions.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R3S3_Reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate weekly Camarilla R3/S3 levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume = df_1d['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily: ADX(14) for range detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all weekly and daily values to daily index
    r3_1w_aligned = align_htf_to_ltf(df_1d['close'].values, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(df_1d['close'].values, df_1w, s3_1w)
    vol_ratio_aligned = align_htf_to_ltf(df_1d['close'].values, df_1d, vol_ratio)
    adx_aligned = align_htf_to_ltf(df_1d['close'].values, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after ADX warmup
        # Get values
        close_val = df_1d['close'].iloc[i]
        r3_1w_val = r3_1w_aligned[i]
        s3_1w_val = s3_1w_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s3_1w_val) or np.isnan(vol_ratio_val) or 
            np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches or goes below S3 with volume confirmation in range market (ADX < 25)
            if (close_val <= s3_1w_val and  # Price at or below S3
                vol_ratio_val > 1.8 and  # Volume confirmation
                adx_val < 25):  # Range market
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above R3 with volume confirmation in range market (ADX < 25)
            elif (close_val >= r3_1w_val and  # Price at or above R3
                  vol_ratio_val > 1.8 and  # Volume confirmation
                  adx_val < 25):  # Range market
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to weekly pivot or shows strength
            pivot_1w_val = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0 if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])) else np.nan
            if not np.isnan(pivot_1w_val) and close_val >= pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to weekly pivot or shows weakness
            pivot_1w_val = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0 if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])) else np.nan
            if not np.isnan(pivot_1w_val) and close_val <= pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals