#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_HTFRegime_v3
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA50 trend filter and 1w ADX regime filter.
In bull markets (price > 1d EMA50): long when price breaks above 12h Camarilla R1 AND 1w ADX > 25.
In bear markets (price < 1d EMA50): short when price breaks below 12h Camarilla S1 AND 1w ADX > 25.
Weekly ADX > 25 ensures we only trade in trending regimes, avoiding choppy markets.
Exit on opposite Camarilla touch or trend reversal. Position size: 0.25.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need to shift by 1 to use previous bar's levels
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    camarilla_range = high_shift - low_shift
    camarilla_r1 = close_shift + camarilla_range * 1.1 / 12
    camarilla_s1 = close_shift - camarilla_range * 1.1 / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (30) and EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Determine weekly ADX regime (trending = ADX > 25)
        weekly_trending = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long setup: price breaks above 12h Camarilla R1 + 1d uptrend + weekly trending regime
            long_setup = (close[i] > camarilla_r1[i]) and htf_1d_bullish and weekly_trending
            
            # Short setup: price breaks below 12h Camarilla S1 + 1d downtrend + weekly trending regime
            short_setup = (close[i] < camarilla_s1[i]) and htf_1d_bearish and weekly_trending
            
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
            # Exit: price touches 12h Camarilla S1 (stop) OR 1d trend turns bearish
            if (close[i] <= camarilla_s1[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 12h Camarilla R1 (stop) OR 1d trend turns bullish
            if (close[i] >= camarilla_r1[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_HTFRegime_v3"
timeframe = "12h"
leverage = 1.0