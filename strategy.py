#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla Pivot Reversal with 1-day Trend Filter and Volume Confirmation.
Long when price touches S1 support in bullish regime (1d EMA34 rising) with volume spike.
Short when price touches R1 resistance in bearish regime (1d EMA34 falling) with volume spike.
Exit when price moves to opposite H1/L1 level or trend reverses.
Camarilla levels provide precise intraday support/resistance; 1d trend ensures alignment with higher timeframe momentum;
volume spike confirms institutional interest. Designed for low trade frequency by requiring confluence of multiple factors.
Works in both bull and bear markets by following the 1d trend direction.
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
    
    # Load 1d data for trend filter and Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Price = (H + L + C) / 3
    # Range = H - L
    tp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    rng = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    H4 = tp + rng * 1.5
    H3 = tp + rng * 1.25
    H2 = tp + rng * 1.1666
    H1 = tp + rng * 1.0833
    L1 = tp - rng * 1.0833
    L2 = tp - rng * 1.1666
    L3 = tp - rng * 1.25
    L4 = tp - rng * 1.5
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1.values)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1.values)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3.values)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price touches S1 (L1) in bullish trend with volume spike
            if (low[i] <= L1_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price touches R1 (H1) in bearish trend with volume spike
            elif (high[i] >= H1_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price moves to opposite H3/L3 level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches H3 or trend turns bearish
                if high[i] >= H3_aligned[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches L3 or trend turns bullish
                if low[i] <= L3_aligned[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0