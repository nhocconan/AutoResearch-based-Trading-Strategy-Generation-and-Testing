#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_Volume_CCIFilter
Hypothesis: Trade breakouts of daily pivot R1/S1 levels on 6h timeframe with volume confirmation and CCI(20) trend filter. 
In bull markets, buy R1 breakouts; in bear markets, sell S1 breakdowns. Uses daily pivot levels as dynamic support/resistance 
that work in both regimes. Volume >1.5x average confirms breakout strength. CCI(20) >0 for longs, <0 for shorts ensures 
trend alignment. Targets 15-30 trades/year to avoid fee drag.
"""

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points: P = (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 6h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # CCI(20) calculation on 6h close
    typical_price = (high + low + close) / 3.0
    cci_period = 20
    cci_ma = np.zeros_like(typical_price)
    cci_mad = np.zeros_like(typical_price)
    
    for i in range(cci_period - 1, len(typical_price)):
        tp_slice = typical_price[i - cci_period + 1:i + 1]
        cci_ma[i] = np.mean(tp_slice)
        cci_mad[i] = np.mean(np.abs(tp_slice - cci_ma[i]))
    
    cci = np.zeros_like(typical_price)
    for i in range(cci_period - 1, len(typical_price)):
        if cci_mad[i] != 0:
            cci[i] = (typical_price[i] - cci_ma[i]) / (0.015 * cci_mad[i])
        else:
            cci[i] = 0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(cci_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if pivot data not available
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above R1 with volume and CCI > 0
            if close[i] > r1_1d_aligned[i] and vol_confirm and cci[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and CCI < 0
            elif close[i] < s1_1d_aligned[i] and vol_confirm and cci[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 or CCI turns negative
            if close[i] < s1_1d_aligned[i] or cci[i] < 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 or CCI turns positive
            if close[i] > r1_1d_aligned[i] or cci[i] > 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_CCIFilter"
timeframe = "6h"
leverage = 1.0