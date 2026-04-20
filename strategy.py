# 4h_Pivot_Band_Scalper
# Hypothesis: Trade mean-reversion within daily pivot bands using 4h candles.
# In both bull and bear markets, price tends to respect daily pivot levels (R1/S1) with mean-reversion.
# Entry: Price touches R1 or S1 and shows rejection (wick) with volume confirmation.
# Exit: Return to pivot (PP) or opposite band.
# Uses tight stops and limited risk per trade to survive volatile markets.
# Target: 20-40 trades/year per symbol with ~55% win rate.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h price action
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if NaN in pivot levels
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_ok = vol > 1.5 * vol_avg
        
        if position == 0:
            # Long setup: price at S1 with rejection (long lower wick) and volume
            at_s1 = abs(price - s1_4h[i]) < 0.001 * s1_4h[i]  # Within 0.1% of S1
            lower_wick = (close[i] - low[i]) > 0.6 * (high[i] - low[i])  # Long lower wick
            if at_s1 and lower_wick and vol_ok:
                signals[i] = 0.25
                position = 1
            
            # Short setup: price at R1 with rejection (long upper wick) and volume
            at_r1 = abs(price - r1_4h[i]) < 0.001 * r1_4h[i]  # Within 0.1% of R1
            upper_wick = (high[i] - close[i]) > 0.6 * (high[i] - low[i])  # Long upper wick
            if at_r1 and upper_wick and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot or stop at S1 breach
            if price >= pivot_4h[i] or price <= s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot or stop at R1 breach
            if price <= pivot_4h[i] or price >= r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Band_Scalper"
timeframe = "4h"
leverage = 1.0