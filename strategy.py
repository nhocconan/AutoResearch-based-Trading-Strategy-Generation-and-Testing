#!/usr/bin/env python3
"""
4h_1d_liquidity_sweep_volume_reversal_v1
Strategy: 4h liquidity sweep detection with volume reversal confirmation
Timeframe: 4h
Leverage: 1.0
Hypothesis: Detects liquidity sweeps (false breakouts of recent highs/lows) followed by volume-confirmed reversals. Uses 1d ATR for dynamic stop placement and 1d ADX for trend strength filtering. Works in both bull and bear markets by capturing mean-reversion after stop hunts. Target: 20-40 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_liquidity_sweep_volume_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d ADX(14) for trend strength
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Strong volume confirmation
    
    # Detect liquidity sweeps: false breakouts of 20-period highs/lows
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Bullish liquidity sweep: price makes new 20-period high but closes below it with volume
    bullish_sweep = (high == high_20) & (close < high_20) & vol_spike
    
    # Bearish liquidity sweep: price makes new 20-period low but closes above it with volume
    bearish_sweep = (low == low_20) & (close > low_20) & vol_spike
    
    # Align sweep signals to avoid look-ahead
    bullish_sweep_aligned = align_htf_to_ltf(prices, df_1d, bullish_sweep.astype(float))
    bearish_sweep_aligned = align_htf_to_ltf(prices, df_1d, bearish_sweep.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade against strong trends (mean reversion in trends)
        strong_trend = adx_1d_aligned[i] > 25
        
        # Mean reversion entries after liquidity sweeps
        long_entry = bullish_sweep_aligned[i] and strong_trend
        short_entry = bearish_sweep_aligned[i] and strong_trend
        
        # Exit when price reverts to the 20-period midpoint or opposite sweep level
        mid_20 = (high_20 + low_20) / 2
        exit_long = position == 1 and close[i] < mid_20[i]
        exit_short = position == -1 and close[i] > mid_20[i]
        
        # Trading logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals