#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Camarilla pivot R1/S1 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 1d Camarilla R1 AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S1 AND volume > 1.5x 20-period average.
Exit when price retraces to 1d Camarilla midpoint (R1+S1)/2 or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by using volume confirmation to filter false breakouts and ATR stops to manage risk.
1d Camarilla levels provide strong institutional support/resistance from higher timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (based on previous day)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    mid = (r1 + s1) / 2.0  # Camarilla midpoint
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    mid_aligned = align_htf_to_ltf(prices, df_1d, mid)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(mid_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        mid_val = mid_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 1d Camarilla R1 AND volume spike
            if (price > r1_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below 1d Camarilla S1 AND volume spike
            elif (price < s1_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to 1d Camarilla midpoint
            if position == 1 and price <= mid_val:
                exit_signal = True
            elif position == -1 and price >= mid_val:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_VolumeConfirmation_ATRStop"
timeframe = "12h"
leverage = 1.0