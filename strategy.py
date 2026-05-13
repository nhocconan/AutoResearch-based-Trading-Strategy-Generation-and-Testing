#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 12h volume spike and 1d ADX trend filter.
# Long when price breaks above R3 AND volume > 2.0x 20-period average AND 1d ADX > 25
# Short when price breaks below S3 AND volume > 2.0x 20-period average AND 1d ADX > 25
# Exit when price reverts to Camarilla pivot (mean reversion) OR ADX < 20 (trend weakens)
# Uses 6h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with volume confirmation and trend filter.
# Camarilla levels provide intraday support/resistance; volume confirms breakout authenticity; ADX filters ranging markets.

name = "6h_Camarilla_R3S3_Breakout_12hVol_1dADX_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 6h data for Camarilla calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for 6h timeframe (using previous bar's range)
    if len(high_6h) >= 2:
        # Use previous 6h bar's high-low for Camarilla calculation
        prev_high = pd.Series(high_6h).shift(1).values
        prev_low = pd.Series(low_6h).shift(1).values
        prev_close = pd.Series(close_6h).shift(1).values
        
        # True range for volatility (using previous bar)
        tr1 = prev_high - prev_low
        tr2 = np.abs(prev_high - np.roll(prev_close, 1))
        tr3 = np.abs(prev_low - np.roll(prev_close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=5, min_periods=5).mean().shift(1).values  # 5-period ATR of previous bars
        
        # Camarilla levels based on previous bar's close and range
        range_val = prev_high - prev_low
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Resistance levels
        r3 = pivot + (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        # Support levels
        s3 = pivot - (range_val * 1.1 / 4)
        s4 = pivot - (range_val * 1.1 / 2)
        
        # For exit: pivot point (mean reversion target)
        camarilla_pivot = pivot
    else:
        r3 = np.full_like(high_6h, np.nan)
        r4 = np.full_like(high_6h, np.nan)
        s3 = np.full_like(high_6h, np.nan)
        s4 = np.full_like(high_6h, np.nan)
        camarilla_pivot = np.full_like(high_6h, np.nan)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pivot)
    
    # Get 12h data for volume spike confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    if len(volume_12h) >= 20:
        vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        volume_filter_12h = volume_12h > (2.0 * vol_ma_12h)
    else:
        vol_ma_12h = np.full_like(volume_12h, np.nan)
        volume_filter_12h = np.full_like(volume_12h, False)
    
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data for trend filter
    if len(high_1d) >= 14:
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        
        # Directional Movement
        up_move = np.diff(high_1d, prepend=high_1d[0])
        down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # invert to positive
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr_1d
        minus_di = 100 * minus_dm_smooth / atr_1d
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    else:
        adx_1d = np.full_like(high_1d, np.nan)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_filter_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > R3 AND volume spike AND strong trend (ADX > 25)
            if close[i] > r3_aligned[i] and volume_filter_12h_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND volume spike AND strong trend (ADX > 25)
            elif close[i] < s3_aligned[i] and volume_filter_12h_aligned[i] and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < pivot (mean reversion) OR trend weakens (ADX < 20)
            if close[i] < pivot_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > pivot (mean reversion) OR trend weakens (ADX < 20)
            if close[i] > pivot_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals