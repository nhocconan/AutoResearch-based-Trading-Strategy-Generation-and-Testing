#!/usr/bin/env python3
"""
6h ADX + Supertrend + Volume Confirmation v1
Hypothesis: ADX filters trending markets (ADX > 25), Supertrend captures direction, and volume confirms breakout strength. This combination avoids whipsaws in sideways markets while capturing sustained moves in both bull and bear regimes by adapting to volatility and trend strength. Targets 15-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_supertrend_volume_v1"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_ma = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d Supertrend (ATR=10, mult=3.0)
    atr_mult = 3.0
    atr_period = 10
    
    # ATR for Supertrend
    tr1_st = high_1d - low_1d
    tr2_st = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_st = np.abs(low_1d - np.roll(close_1d, 1))
    tr_st = np.maximum(tr1_st, np.maximum(tr2_st, tr3_st))
    tr_st[0] = high_1d[0] - low_1d[0]
    atr_st = pd.Series(tr_st).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic bands
    basic_ub = (high_1d + low_1d) / 2 + atr_mult * atr_st
    basic_lb = (high_1d + low_1d) / 2 - atr_mult * atr_st
    
    # Final bands
    final_ub = np.zeros_like(close_1d)
    final_lb = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # Align 1d indicators to 6s
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # 6s ATR(10) for volume filter volatility adjustment
    tr6 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr6[0] = high[0] - low[0]
    atr6 = pd.Series(tr6).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume filter (>2.0x ATR-adjusted average)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * (atr6 / np.maximum(atr6.mean(), 0.001)))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend weakens or Supertrend flips
            if adx_aligned[i] < 20 or close[i] < supertrend_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens or Supertrend flips
            if adx_aligned[i] < 20 or close[i] > supertrend_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: strong trend + price above Supertrend + volume
            if (adx_aligned[i] > 25 and 
                close[i] > supertrend_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: strong trend + price below Supertrend + volume
            elif (adx_aligned[i] > 25 and 
                  close[i] < supertrend_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals