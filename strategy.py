#!/usr/bin/env python3
# 4h_Vortex_Trend_Reversal_Volume
# Hypothesis: Vortex indicator detects trend reversals; long when VI+ > VI- and price above 1d EMA34, short when VI- > VI+ and price below EMA34, with volume confirmation (>1.5x 20-period average). Exit on Vortex crossover. Designed for 25-40 trades/year with clear trend and volume to avoid false signals. Works in bull via trend continuation and bear via reversals at extremes.

name = "4h_Vortex_Trend_Reversal_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Vortex and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex indicator (VI+ and VI-) on 1d data
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 
                                   np.abs(high_1d[0] - close_1d[0]), 
                                   np.abs(low_1d[0] - close_1d[0])])], tr1])
    
    vm_plus = np.abs(high_1d[1:] - low_1d[:-1])
    vm_minus = np.abs(low_1d[1:] - high_1d[:-1])
    vm_plus = np.concatenate([[high_1d[0] - low_1d[0]], vm_plus])
    vm_minus = np.concatenate([[high_1d[0] - low_1d[0]], vm_minus])
    
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / \
              pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / \
               pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Align Vortex indicators to 4h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vi_plus_val = vi_plus_aligned[i]
        vi_minus_val = vi_minus_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        if position == 0:
            # LONG: VI+ > VI- (bullish trend) and price above EMA34 with volume confirmation
            if vi_plus_val > vi_minus_val and close_1d_current > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish trend) and price below EMA34 with volume confirmation
            elif vi_minus_val > vi_plus_val and close_1d_current < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- crosses above VI+ (trend reversal to bearish)
            if vi_minus_val > vi_plus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ crosses above VI- (trend reversal to bullish)
            if vi_plus_val > vi_minus_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals