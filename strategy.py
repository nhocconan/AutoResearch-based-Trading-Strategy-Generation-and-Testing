#!/usr/bin/env python3
"""
12h Fractal Breakout with 1d Trend and Volume Confirmation v1
Hypothesis: Price breaks above/below Williams fractal levels on 12h, 
aligned with strong daily trend (ADX>25) and volume spikes (>2x 20-period average),
captures momentum moves while avoiding false breakouts. Works in bull/bear by 
requiring trend alignment. Target: 12-37 trades/year per symbol.
"""

name = "12h_fractal_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ADX for 1d
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Williams fractals on 12h price
    bearish_fractal, bullish_fractal = compute_williams_fractals(high, low)
    # Fractals need 2 extra 12h bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_ltf_to_htf(prices, bearish_fractal)
    bullish_fractal_aligned = align_ltf_to_htf(prices, bullish_fractal)
    
    # Volume spike detector: current volume > 2 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d ADX for current 12h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Regime filter: only trade in strong trending markets on 1d
        strong_trend_1d = adx_1d_aligned > 25
        
        if position == 1:  # Long position
            # Exit: trend weakens OR price closes below bullish fractal
            if not strong_trend_1d or close[i] < bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens OR price closes above bearish fractal
            if not strong_trend_1d or close[i] > bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume spike and strong 1d trend
            # Breakout conditions: price breaks fractal levels
            if volume_spike[i] and strong_trend_1d and close[i] > bullish_fractal_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif volume_spike[i] and strong_trend_1d and close[i] < bearish_fractal_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals