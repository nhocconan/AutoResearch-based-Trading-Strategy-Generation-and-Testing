#!/usr/bin/env python3
"""
6h_HTF_Pivot_Momentum_Confluence
Hypothesis: Combines 1d pivot point support/resistance with 1w EMA trend and volume confirmation.
Looks for price rejection at pivot levels (S1/S2/R1/R2) with momentum confirmation in the direction
of the weekly trend. Designed to work in both bull and bear markets by fading extremes in ranging
markets and following momentum in trending markets. Targets 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 21-period EMA on weekly close
    ema21_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema21_1w[20] = np.mean(close_1w[0:21])
        alpha = 2 / (21 + 1)
        for i in range(21, len(close_1w)):
            ema21_1w[i] = close_1w[i] * alpha + ema21_1w[i-1] * (1 - alpha)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align 1d pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Align 1w EMA to 6h timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on weekly EMA
        is_uptrend = close[i] > ema21_1w_aligned[i]
        
        if position == 0:
            # Long conditions: price near support with bullish momentum
            near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005  # Within 0.5% of S1
            near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < 0.005  # Within 0.5% of S2
            
            # Short conditions: price near resistance with bearish momentum
            near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005  # Within 0.5% of R1
            near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < 0.005  # Within 0.5% of R2
            
            if (near_s1 or near_s2) and vol_spike[i]:
                # In uptrend, look for bounce; in downtrend, look for capitulation bounce
                if is_uptrend or (not is_uptrend and close[i] > open_prices[i]):
                    signals[i] = 0.25
                    position = 1
            elif (near_r1 or near_r2) and vol_spike[i]:
                # In downtrend, look for rejection; in uptrend, look for exhaustion
                if not is_uptrend or (is_uptrend and close[i] < open_prices[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or shows weakness
            if close[i] >= pivot_aligned[i] or (close[i] < open_prices[i] and vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or shows strength
            if close[i] <= pivot_aligned[i] or (close[i] > open_prices[i] and vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HTF_Pivot_Momentum_Confluence"
timeframe = "6h"
leverage = 1.0