#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime
# Elder Ray (bull_power = high - EMA13, bear_power = EMA13 - low) measures bull/bear strength.
# 12h ADX > 25 confirms trending regime; ADX < 20 indicates ranging.
# Long when bull_power > 0 and ADX > 25; short when bear_power > 0 and ADX > 25.
# Exit when ADX < 20 (range) or power reverses.
# Designed to capture trends in both bull and bear markets while avoiding whipsaws in ranges.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 12h ADX for regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i])):
            continue
        
        # ADX regime: >25 = trend, <20 = range
        if adx_12h_aligned[i] > 25:
            # Trending regime: follow Elder Ray
            if bull_power[i] > 0:
                signals[i] = 0.25
            elif bear_power[i] > 0:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1] if i > 0 else 0
        elif adx_12h_aligned[i] < 20:
            # Ranging regime: fade extreme power (optional, but we stay flat to avoid whipsaw)
            signals[i] = 0.0
        else:
            # Transition zone: hold previous
            signals[i] = signals[i-1] if i > 0 else 0
    
    return signals

name = "6h_ElderRay_12hADX_Regime"
timeframe = "6h"
leverage = 1.0