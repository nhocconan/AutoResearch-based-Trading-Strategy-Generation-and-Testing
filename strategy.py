#!/usr/bin/env python3
"""
6h Elder Ray Power with 1d ADX Regime Filter
Long when Bull Power > 0 and Bear Power < 0 in bullish regime (ADX > 25 and +DI > -DI)
Short when Bear Power > 0 and Bull Power < 0 in bearish regime (ADX > 25 and -DI > +DI)
Exit when power signals weaken or regime changes
Designed to capture institutional buying/selling pressure with trend confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_adx_regime_v1"
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
    
    # === Elder Ray Power (13-period EMA) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d ADX/DI Regime Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First value is NaN
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM (14-period)
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # Calculate DI and ADX
    plus_di = 100 * plus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align to 6t timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power weakens OR regime turns bearish
            if bull_power_aligned[i] <= 0 or (adx_aligned[i] > 25 and minus_di_aligned[i] > plus_di_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power weakens OR regime turns bullish
            if bear_power_aligned[i] >= 0 or (adx_aligned[i] > 25 and plus_di_aligned[i] > minus_di_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need strong trend (ADX > 25) for entry
            if adx_aligned[i] > 25:
                # Bullish regime: +DI > -DI
                if plus_di_aligned[i] > minus_di_aligned[i]:
                    # Look for Bull Power > 0 and Bear Power < 0 (strong buying)
                    if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0:
                        position = 1
                        signals[i] = 0.25
                # Bearish regime: -DI > +DI
                elif minus_di_aligned[i] > plus_di_aligned[i]:
                    # Look for Bear Power > 0 and Bull Power < 0 (strong selling)
                    if bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0:
                        position = -1
                        signals[i] = -0.25
    
    return signals