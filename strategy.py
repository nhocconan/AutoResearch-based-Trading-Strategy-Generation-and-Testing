#!/usr/bin/env python3
"""
6h_1d_Adaptive_Range_Breakout_with_Volume_and_ADX
Hypothesis: In ranging markets (ADX < 25), price tends to revert from daily support/resistance; 
in trending markets (ADX >= 25), breakouts of daily high/low with volume continuation are traded.
This adapts to both bull and bear regimes by using volatility-based position sizing and ADX regime filter.
Target: 15-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for range and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility normalization
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily range: (high - low) / ATR
    daily_range = (high_1d - low_1d) / atr_1d
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    
    # Calculate ADX(14) on daily data to determine regime
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    atr_1d_for_adx = atr_1d  # already calculated
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values  # same as atr_1d
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_1d_smooth
    minus_di = 100 * minus_dm_smooth / atr_1d_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) != 0, dx, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Daily high and low for breakout levels
    daily_high = high_1d
    daily_low = low_1d
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    base_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(daily_range_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX < 25 = ranging, ADX >= 25 = trending
        ranging = adx_aligned[i] < 25
        trending = adx_aligned[i] >= 25
        
        # Volatility-adjusted position size: smaller in high volatility
        vol_factor = np.clip(1.0 / (daily_range_aligned[i] / 100), 0.5, 1.5)  # normalize around 100% ATR range
        position_size = base_size * vol_factor
        
        if ranging:
            # Mean reversion: fade at daily extremes
            # Long near daily low, short near daily high
            long_condition = (close[i] <= daily_low_aligned[i] * 1.001) and volume_expansion[i]
            short_condition = (close[i] >= daily_high_aligned[i] * 0.999) and volume_expansion[i]
        else:
            # Trending: breakout continuation
            # Long on daily high break, short on daily low break
            long_condition = (close[i] > daily_high_aligned[i]) and volume_expansion[i]
            short_condition = (close[i] < daily_low_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Adaptive_Range_Breakout_with_Volume_and_ADX"
timeframe = "6h"
leverage = 1.0