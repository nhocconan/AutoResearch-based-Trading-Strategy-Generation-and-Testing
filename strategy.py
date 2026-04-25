#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend_RegimeFilter
Hypothesis: Use 6h Elder Ray (Bull/Bear Power) with 1d EMA13 for trend strength and 1w ADX for regime filtering.
In trending markets (ADX > 25): go long when Bull Power > 0 and EMA13 rising, short when Bear Power < 0 and EMA13 falling.
In ranging markets (ADX <= 25): fade extremes - long when Bull Power < -0.5*ATR and price > 1d VWAP, short when Bear Power > 0.5*ATR and price < 1d VWAP.
Volume confirmation required for all entries. Position size: 0.25. Target: 80-120 total trades over 4 years.
"""

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
    
    # Get 1d data for HTF trend and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for HTF regime (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ATR(14) for Elder Ray scaling
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d VWAP (approximation using typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (pd.Series(typical_price_1d * df_1d['volume'].values).rolling(window=20, min_periods=20).sum() / 
               pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).sum()).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 1w ADX(14) for regime filtering
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w, 1)), np.absolute(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus_1w = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                          np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus_1w = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                           np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus_1w[0] = 0
    dm_minus_1w[0] = 0
    
    # Smoothed TR, DM+
    tr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    dm_plus_14_1w = pd.Series(dm_plus_1w).rolling(window=14, min_periods=14).mean().values
    dm_minus_14_1w = pd.Series(dm_minus_1w).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14_1w = 100 * dm_plus_14_1w / tr_14_1w
    di_minus_14_1w = 100 * dm_minus_14_1w / tr_14_1w
    
    # DX and ADX
    dx_14_1w = 100 * np.absolute(di_plus_14_1w - di_minus_14_1w) / (di_plus_14_1w + di_minus_14_1w)
    adx_14_1w = pd.Series(dx_14_1w).rolling(window=14, min_periods=14).mean().values
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate 6h Elder Ray components
    # Bull Power = High - EMA13(close)
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h
    
    # Bear Power = Low - EMA13(close)
    bear_power = low - ema_13_6h
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13), ATR (14), volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or
            np.isnan(adx_14_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d trend (EMA13 slope)
        if i >= 2:
            ema13_rising = ema_13_1d_aligned[i] > ema_13_1d_aligned[i-1]
            ema13_falling = ema_13_1d_aligned[i] < ema_13_1d_aligned[i-1]
        else:
            ema13_rising = False
            ema13_falling = False
        
        # Regime filter: 1w ADX > 25 = trending, <= 25 = ranging
        is_trending = adx_14_1w_aligned[i] > 25
        is_ranging = adx_14_1w_aligned[i] <= 25
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            if is_trending and volume_confirm:
                # Trending market: follow Elder Ray with trend
                long_setup = (bull_power[i] > 0) and ema13_rising
                short_setup = (bear_power[i] < 0) and ema13_falling
            elif is_ranging and volume_confirm:
                # Ranging market: fade extremes relative to VWAP
                long_setup = (bull_power[i] < -0.5 * atr_14_1d_aligned[i]) and (close[i] > vwap_1d_aligned[i])
                short_setup = (bear_power[i] > 0.5 * atr_14_1d_aligned[i]) and (close[i] < vwap_1d_aligned[i])
            else:
                long_setup = False
                short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: trend exhaustion or regime change
            if is_trending:
                exit_condition = (bull_power[i] <= 0) or (not ema13_rising)
            else:
                exit_condition = (bull_power[i] >= -0.25 * atr_14_1d_aligned[i]) or (close[i] < vwap_1d_aligned[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: trend exhaustion or regime change
            if is_trending:
                exit_condition = (bear_power[i] >= 0) or (not ema13_falling)
            else:
                exit_condition = (bear_power[i] <= 0.25 * atr_14_1d_aligned[i]) or (close[i] > vwap_1d_aligned[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0