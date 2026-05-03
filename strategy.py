#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Bull power = high - EMA13(close), Bear power = EMA13(close) - low
# Long when Bull power > 0 AND Bear power < 0 AND 1d ADX > 25 (strong trend)
# Short when Bear power > 0 AND Bull power < 0 AND 1d ADX > 25 (strong trend)
# Uses 6h primary timeframe with 1d HTF for ADX regime filter and EMA13 for Elder Ray.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Elder Ray measures bull/bear strength relative to trend EMA; ADX filters weak/choppy markets.

name = "6h_ElderRay_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 1d ADX for regime filter (trend strength)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, min_periods=tr_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull/Bear power
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Elder Ray signals
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] > 0
        
        # Entry logic
        if position == 0:
            if strong_trend and bull_strong and not bear_strong:
                signals[i] = 0.25
                position = 1
            elif strong_trend and bear_strong and not bull_strong:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakens OR bear power dominates
            if not strong_trend or bear_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend weakens OR bull power dominates
            if not strong_trend or bull_strong:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals