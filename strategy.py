#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Pullback_Strategy
Timeframe: 6h
Primary Signal: Pullback to weekly pivot levels with volume confirmation and 1d trend filter
Logic:
- Weekly pivot (H+L+C)/3 from prior week acts as support/resistance
- In uptrend (price > 1d EMA50), go long on pullback to weekly S1 with volume spike
- In downtrend (price < 1d EMA50), go short on pullback to weekly R1 with volume spike
- Exit when price moves back to weekly pivot or trend changes
Designed for low-frequency, high-conviction trades in both bull and bear markets
"""

name = "6h_Weekly_Pivot_Pullback_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Support and resistance levels
    R1 = 2 * weekly_pivot - df_1w['low']
    S1 = 2 * weekly_pivot - df_1w['high']
    
    # Align weekly pivot levels to 6h timeframe (waits for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average (rare but significant)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for EMA50 and weekly alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > 1d EMA50 (uptrend) + pullback to weekly S1 + volume spike
            if (close[i] > ema50[i] and 
                low[i] <= S1_aligned[i] * 1.005 and  # Allow 0.5% tolerance for pullback
                close[i] > S1_aligned[i] and        # Close above support (bullish rejection)
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < 1d EMA50 (downtrend) + pullback to weekly R1 + volume spike
            elif (close[i] < ema50[i] and 
                  high[i] >= R1_aligned[i] * 0.995 and  # Allow 0.5% tolerance for pullback
                  close[i] < R1_aligned[i] and          # Close below resistance (bearish rejection)
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot OR trend turns bearish
            if (close[i] >= pivot_aligned[i] * 0.995) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot OR trend turns bullish
            if (close[i] <= pivot_aligned[i] * 1.005) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals