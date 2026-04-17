#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams Fractal breakouts + volume confirmation + ADX trend filter.
Long when price breaks above recent bearish fractal with volume spike and ADX>25.
Short when price breaks below recent bullish fractal with volume spike and ADX>25.
Williams Fractals provide reliable support/resistance levels that work in both bull and bear markets.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams Fractals
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 4h (Williams Fractals need 2-bar confirmation delay)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume spike (20-bar volume ratio)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = volume_ratio[i] > 1.5
        
        # Trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above bearish fractal resistance with volume and trend
            if (close[i] > bearish_fractal_aligned[i] and 
                volume_confirm and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal support with volume and trend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  volume_confirm and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below bullish fractal support or trend weakens
            if (close[i] < bullish_fractal_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above bearish fractal resistance or trend weakens
            if (close[i] > bearish_fractal_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsFractal_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0