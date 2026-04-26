#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeRegime
Hypothesis: Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume regime confirmation.
Only trade breakouts aligned with 4h trend during low volatility regimes (Choppiness Index > 61.8).
Discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year per symbol.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Volume regime filter reduces whipsaw in ranging markets.
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
    
    # Get 4h data for EMA50 trend filter and Choppiness Index
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Choppiness Index (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 4h bars
    
    # ATR14
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    # Avoid division by zero
    range_14 = hh14 - ll14
    chop_raw = np.where(range_14 > 0, sum_tr14 / range_14, np.nan)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d_prev + rang * 1.1 / 12
    s1 = close_1d_prev - rang * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter (4h EMA50)
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume regime: only trade in low volatility (choppy) markets
        low_vol_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in uptrend AND low volatility regime
            if close[i] > r1_aligned[i] and volume_spike[i] and uptrend and low_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in downtrend AND low volatility regime
            elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend and low_vol_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R1 OR trend changes OR high volatility regime
            if close[i] < r1_aligned[i] or not uptrend or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S1 OR trend changes OR high volatility regime
            if close[i] > s1_aligned[i] or not downtrend or chop_aligned[i] <= 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeRegime"
timeframe = "4h"
leverage = 1.0