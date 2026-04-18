#!/usr/bin/env python3
"""
12h Daily Pivot R1/S1 Breakout with Volume Spike and ATR Stop
Hypothesis: Daily pivot levels (R1, S1) act as key support/resistance. Breakouts beyond these levels with volume confirmation capture momentum in both bull and bear markets. Using 12h timeframe reduces trade frequency to minimize fee drag, while daily pivots provide structure. Volume spike filter ensures breakouts have conviction. ATR-based stop loss manages risk.
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
    
    # Get daily data for pivot calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points using standard formula
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous day's data to avoid look-ahead
    daily_high = df_d['high']
    daily_low = df_d['low']
    daily_close = df_d['close']
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Shift by 1 to use previous day's levels only
    r1_prev = r1.shift(1).values
    s1_prev = s1.shift(1).values
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1_prev)
    
    # ATR for stop loss and volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike
            if price > r1_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 with volume spike
            elif price < s1_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Stop loss: 2.5 * ATR below entry
            if price <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Stop loss: 2.5 * ATR above entry
            if price >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DailyPivot_R1S1_Breakout_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0