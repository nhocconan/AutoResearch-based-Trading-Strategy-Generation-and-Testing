#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop
Hypothesis: Trade 4h Camarilla R1/S1 breakouts in direction of 1d EMA34 trend with volume confirmation and chop filter.
Uses 4h primary timeframe for optimal trade frequency (19-50/year target) and 1d trend/chop filters for robustness.
Camarilla levels from prior 4h provide structure; 1d EMA34 filters trend direction; 1d chop regime avoids whipsaw in ranging markets;
volume spike on 4h confirms breakout conviction. Works in bull/bear via trend filter + volume + chop confluence.
Target: 25-40 trades/year per symbol to balance opportunity and fee drag.
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chop regime (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    atr_1d = np.zeros_like(close_1d_arr)
    for i in range(1, len(close_1d_arr)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d_arr[i-1]), abs(low_1d[i] - close_1d_arr[i-1]))
        atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14 if i > 1 else tr
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_max_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_min_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.maximum(high_max_14 - low_min_14, 1e-10)
    chop = 100 * np.log10(atr_sum_14 / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Camarilla levels for today (based on prior 4h OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_range = (high_4h - low_4h) * 1.1 / 12.0
    camarilla_R1 = close_4h + camarilla_range * 1.0  # R1 = close + 1*(range)
    camarilla_S1 = close_4h - camarilla_range * 1.0  # S1 = close - 1*(range)
    
    # Align Camarilla levels to 4h timeframe (prior 4h's levels available at 4h close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    
    # Volume confirmation: volume > 1.8x 20-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), chop (34), volume MA (20), aligned indicators
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
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
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above 1d EMA34 + volume spike + trending regime
            long_breakout = close[i] > camarilla_R1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i] and trending_regime
            
            # Short: price breaks below Camarilla S1 + price below 1d EMA34 + volume spike + trending regime
            short_breakout = close[i] < camarilla_S1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i] and trending_regime
            
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
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA) OR chop > 61.8 (ranging)
            if (close[i] < camarilla_S1_aligned[i] or not price_above_ema or chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA) OR chop > 61.8 (ranging)
            if (close[i] > camarilla_R1_aligned[i] or not price_below_ema or chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeChop"
timeframe = "4h"
leverage = 1.0