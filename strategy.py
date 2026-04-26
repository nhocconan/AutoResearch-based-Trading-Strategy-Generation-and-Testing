#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Trade 12h Camarilla R1/S1 breakouts in direction of 1d EMA50 trend with volume confirmation and chop filter.
Uses 12h primary timeframe for lower trade frequency (target: 12-37/year) and 1d trend filter for robustness.
Camarilla levels from prior 12h provide structure; 1d EMA50 filters for higher timeframe trend alignment; 
volume spike on 12h confirms breakout; choppiness index avoids whipsaws in ranging markets.
Works in bull/bear via trend filter + volume confirmation + regime filter.
Target: 50-150 total trades over 4 years to avoid fee drag.
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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1d data for choppiness index (regime filter)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(tr_sum_14 / range_14) / np.log10(14), 50)
    chop = np.nan_to_num(chop, nan=50.0)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels for today (based on prior 12h OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    camarilla_R1 = close_12h + camarilla_range * 1.0  # R1 = close + 1*(range)
    camarilla_S1 = close_12h - camarilla_range * 1.0  # S1 = close - 1*(range)
    
    # Align Camarilla levels to 12h timeframe (prior 12h's levels available at 12h close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S1)
    
    # Volume confirmation: volume > 2.0x 20-period average on 12h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), chop (14), volume MA (20), aligned indicators
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: choppiness index (avoid extremes)
        # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        # We'll use CHOP < 61.8 to avoid strong ranging markets where breakouts fail
        not_ranging = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 1d EMA50 + volume spike + not ranging
            long_breakout = close[i] > camarilla_R1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i] and not_ranging
            
            # Short: price breaks below Camarilla S1 + price below 1d EMA50 + volume spike + not ranging
            short_breakout = close[i] < camarilla_S1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i] and not_ranging
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA) OR strong ranging
            if (close[i] < camarilla_S1_aligned[i] or not price_above_ema or chop_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA) OR strong ranging
            if (close[i] > camarilla_R1_aligned[i] or not price_below_ema or chop_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0