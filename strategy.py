#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_R2S2_Breakout_Volume_Filter
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1, R2/S2) with volume confirmation and 1d EMA34 trend filter. 
Breakouts beyond R2/S2 indicate strong momentum, while bounces at R1/S1 with volume capture reversals. 
Designed for 6h timeframe to limit trades (12-37/year) and work in both bull/bear markets by following higher-timeframe trend.
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
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        k = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = close_1d[i] * k + ema_34_1d[i-1] * (1 - k)
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # R2 = close + 0.55*(high-low)
    # R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low)
    # S2 = close - 0.55*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    camarilla_r2 = np.full(len(close_1d), np.nan)
    camarilla_s2 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            continue  # Skip first day as we need previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        diff = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + 0.275 * diff
        camarilla_s1[i] = prev_close - 0.275 * diff
        camarilla_r2[i] = prev_close + 0.55 * diff
        camarilla_s2[i] = prev_close - 0.55 * diff
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 35  # Warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long breakout: price breaks above R2 with volume and above EMA
            if close[i] > r2_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Long reversal: price bounces above S1 with volume and above EMA
            elif close[i] > s1_aligned[i] and vol_spike[i] and close[i] > ema_34_aligned[i] and close[i] < r1_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short breakout: price breaks below S2 with volume and below EMA
            elif close[i] < s2_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            # Short reversal: price bounces below R1 with volume and below EMA
            elif close[i] < r1_aligned[i] and vol_spike[i] and close[i] < ema_34_aligned[i] and close[i] > s1_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit conditions: hold minimum 2 bars, then exit on reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] < ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit conditions: hold minimum 2 bars, then exit on reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] > ema_34_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "6h_Camarilla_R1S1_R2S2_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0