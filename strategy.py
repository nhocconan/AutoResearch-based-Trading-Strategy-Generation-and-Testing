#!/usr/bin/env python3
"""
1d_WeeklyPivotResistanceBreakout
Hypothesis: Break above weekly pivot resistance on daily close with volume confirmation signals strong bullish momentum. 
Exit on close below weekly pivot support or loss of bullish momentum. 
Designed for low trade frequency to avoid fee drag while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly high, low, close for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot and resistance/support levels
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w      # Resistance 1
    s1 = 2 * pivot - high_1w     # Support 1
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume spike: >1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14*2)  # Need warmup for volume MA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_spike = volume_spike[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: close above weekly R1 with volume spike and bullish momentum (RSI > 50)
            if price > r1_val and vol_spike and rsi_val > 50:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: close below weekly S1 OR loss of bullish momentum (RSI < 40)
            if price < s1_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivotResistanceBreakout"
timeframe = "1d"
leverage = 1.0