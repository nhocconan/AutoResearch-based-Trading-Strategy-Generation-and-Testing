#!/usr/bin/env python3
"""
6h Camarilla Pivot from 1d with Volume Spike and Momentum Filter
Hypothesis: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide institutional support/resistance. In ranging markets (ADX<25), fade extremes 
at R3/S3. In trending markets (ADX>25), breakout continuations at R4/S4. 
Volume confirms participation. Works in bull/bear by adapting to regime.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # ADX for regime detection (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr = np.maximum(high - low, 
                    np.maximum(np.abs(high - np.roll(close, 1)), 
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get Camarilla pivot levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r3 = close_1d + range_1d * 1.1 / 4
    r4 = close_1d + range_1d * 1.1 / 2
    # Support levels
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below pivot OR ADX drops (trend weakening)
            if close[i] < pivot_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above pivot OR ADX drops
            if close[i] > pivot_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Regime-based logic
            if adx[i] > 25:  # Trending market - breakout continuation
                # Long breakout: close above R4 with volume
                if close[i] > r4_aligned[i] and vol_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: close below S4 with volume
                elif close[i] < s4_aligned[i] and vol_spike[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging market - mean reversion at extremes
                # Long mean reversion: bounce from S3 with volume
                if close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and vol_spike[i]:
                    position = 1
                    signals[i] = 0.25
                # Short mean reversion: rejection at R3 with volume
                elif close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and vol_spike[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals