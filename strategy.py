# 6H_Camarilla1d_1wTrend_Volume_Strategy
# Hypothesis: Use 1-day Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) with 1-week trend filter and volume confirmation.
# In trending markets (ADX > 25 on weekly), breakouts at R4/S4 continue the trend.
# In ranging markets (ADX < 25 on weekly), reversals at R3/S3 capture mean reversion.
# This adapts to both bull and bear markets by using weekly trend to determine Camarilla interpretation.
# Targets 15-35 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivots - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d + (range_ * 1.1 / 12)
    r2 = close_1d + (range_ * 1.1 / 6)
    r3 = close_1d + (range_ * 1.1 / 4)
    r4 = close_1d + (range_ * 1.1 / 2)
    
    # Support levels
    s1 = close_1d - (range_ * 1.1 / 12)
    s2 = close_1d - (range_ * 1.1 / 6)
    s3 = close_1d - (range_ * 1.1 / 4)
    s4 = close_1d - (range_ * 1.1 / 2)
    
    # Load 1-week data for trend filter (ADX) - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1-week ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
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
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Align HTF indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Volume average (50-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        r4_val = r4_aligned[i]
        s3_val = s3_aligned[i]
        s4_val = s4_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Determine market regime based on weekly ADX
            if adx_val > 25:  # Trending market
                # Breakout continuation: Go with the trend
                if close_val > r4_val and vol_current > 1.5 * vol_ma_val:
                    signals[i] = 0.25
                    position = 1
                elif close_val < s4_val and vol_current > 1.5 * vol_ma_val:
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging market
                # Mean reversion at R3/S3
                if close_val < r3_val and vol_current > 1.5 * vol_ma_val:
                    signals[i] = 0.25  # Long at S3 (price < R3 suggests bounce)
                    position = 1
                elif close_val > s3_val and vol_current > 1.5 * vol_ma_val:
                    signals[i] = -0.25  # Short at R3 (price > S3 suggests rejection)
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches opposite Camarilla level or trend changes
                if adx_val > 25:  # Still trending
                    if close_val >= r4_val:  # Reached resistance target
                        exit_signal = True
                else:  # Now ranging
                    if close_val <= s3_val:  # Reached support
                        exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches opposite Camarilla level or trend changes
                if adx_val > 25:  # Still trending
                    if close_val <= s4_val:  # Reached support target
                        exit_signal = True
                else:  # Now ranging
                    if close_val >= r3_val:  # Reached resistance
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla1d_1wTrend_Volume_Strategy"
timeframe = "6h"
leverage = 1.0