#!/usr/bin/env python3
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Shift to use previous week's pivots (avoid look-ahead)
    r1_1w_prev = np.roll(r1_1w, 1)
    s1_1w_prev = np.roll(s1_1w, 1)
    r2_1w_prev = np.roll(r2_1w, 1)
    s2_1w_prev = np.roll(s2_1w, 1)
    r1_1w_prev[0] = np.nan
    s1_1w_prev[0] = np.nan
    r2_1w_prev[0] = np.nan
    s2_1w_prev[0] = np.nan
    
    # Align weekly pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, r1_1w_prev)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1_1w_prev)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2_1w_prev)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2_1w_prev)
    
    # Volume confirmation: current volume > 1.5 * 4-period average (12h * 4 = 48h ~ 2d)
    volume_ma4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need R2/S2 and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma4[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(r2_12h[i]) or 
            np.isnan(s2_12h[i]) or
            np.isnan(r1_12h[i]) or 
            np.isnan(s1_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 4-period average
        volume_filter = volume[i] > (1.5 * volume_ma4[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume and volatility (strong breakout)
            if close[i] > r2_12h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume and volatility (strong breakdown)
            elif close[i] < s2_12h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_12h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_12h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R2_S2_Breakout_Vol_v1"
timeframe = "12h"
leverage = 1.0