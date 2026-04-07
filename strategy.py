#!/usr/bin/env python3
"""
6h_momentum_pullback_v1
Hypothesis: On 6h timeframe, enter long when price pulls back to 6h EMA21 during a strong 12h uptrend (ADX>25) with rising volume, enter short when price rallies to 6h EMA21 during a strong 12h downtrend (ADX>25) with rising volume. Exit when price closes beyond EMA21 or ADX weakens (<20). Designed for 15-25 trades/year to minimize fee dust while capturing trend continuation moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_momentum_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h ADX for trend strength filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum.reduce([tr1, tr2, tr3])])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, ignore_na=False).mean().values
    
    # Align 12h indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(ema_21[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(close[i-1])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA21 or ADX weakens
            if close[i] < ema_21[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA21 or ADX weakens
            if close[i] > ema_21[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and adx_aligned[i] >= 25:
                # Long: pullback to EMA21 in uptrend
                if close[i] <= ema_21[i] and close[i-1] > ema_21[i-1] and close[i] > close[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: rally to EMA21 in downtrend
                elif close[i] >= ema_21[i] and close[i-1] < ema_21[i-1] and close[i] < close[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals