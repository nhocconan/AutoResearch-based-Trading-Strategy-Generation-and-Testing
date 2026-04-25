#!/usr/bin/env python3
"""
6h_ElderRay_WeeklyTrend_RegimeFilter_v1
Hypothesis: Trade Elder Ray (Bull/Bear Power) on 6h with weekly trend filter and ADX regime filter. In bullish weekly trend (price > weekly EMA50), take long signals when Bull Power > 0 and rising. In bearish weekly trend (price < weekly EMA50), take short signals when Bear Power < 0 and falling. Uses ADX > 20 to confirm trending market and avoid chop. Designed for 6h timeframe with moderate trade frequency (~20-40/year) to balance opportunity and fee drag while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Elder Ray on 6h timeframe
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 and ADX
    start_idx = max(13, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly HTF trend
        weekly_bullish = close[i] > ema_50_1w_aligned[i]
        weekly_bearish = close[i] < ema_50_1w_aligned[i]
        
        # ADX regime filter: only trade in trending markets (ADX > 20)
        trending_market = adx_aligned[i] > 20
        
        if position == 0:
            # Look for Elder Ray signals with regime filter
            long_signal = (bull_power[i] > 0) and (bull_power[i] > bull_power[i-1])  # rising bull power
            short_signal = (bear_power[i] < 0) and (bear_power[i] < bear_power[i-1])  # falling bear power
            
            # Only trade in direction of weekly trend with trend confirmation
            if long_signal and weekly_bullish and trending_market:
                signals[i] = 0.25
                position = 1
            elif short_signal and weekly_bearish and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when bull power turns negative or weekly trend turns bearish
            exit_signal = (bull_power[i] <= 0) or (not weekly_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when bear power turns positive or weekly trend turns bullish
            exit_signal = (bear_power[i] >= 0) or weekly_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_WeeklyTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0