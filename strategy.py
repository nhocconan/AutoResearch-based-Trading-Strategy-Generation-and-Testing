#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Confirmation_v5
Hypothesis: On 12h timeframe, use daily Camarilla pivot levels (R1/S1) as dynamic support/resistance.
Long when price breaks above R1 with volume confirmation; short when breaks below S1 with volume confirmation.
Uses daily volatility filter (ATR ratio) to avoid choppy markets. Designed for low trade frequency (15-25/year)
to minimize fee drag while capturing meaningful intraday moves in both bull and bear markets.
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
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily ATR for volatility filter (20-period)
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]  # First period
    atr_20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20_1d)
    
    # Current ATR ratio (ATR / prior ATR) to detect volatility expansion
    atr_ratio = atr_20_1d / np.roll(atr_20_1d, 1)
    atr_ratio[0] = 1.0
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        vol_expansion = atr_ratio_aligned[i] > 1.2  # Volatility expanding
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility expansion
            if price > r1 and vol_spike and vol_expansion:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility expansion
            elif price < s1 and vol_spike and vol_expansion:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below R1 or volatility contracts
            if price < r1 or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above S1 or volatility contracts
            if price > s1 or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Confirmation_v5"
timeframe = "12h"
leverage = 1.0