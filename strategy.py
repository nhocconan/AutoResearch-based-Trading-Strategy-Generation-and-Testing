#!/usr/bin/env python3
# Hypothesis: 12h price action near 1d pivot levels with volume confirmation and trend filter
# Long when price is above daily pivot, above 12h EMA50, and volume > 1.5x 20-period average
# Short when price is below daily pivot, below 12h EMA50, and volume > 1.5x 20-period average
# Exit when price crosses back below/above pivot OR EMA direction contradicts position
# Position size: 0.28 (28% of capital) to balance return and drawdown
# Designed to work in trending markets via EMA filter and in ranging markets via pivot reversals

name = "12h_Pivot_EMA_Volume_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for pivot points (daily high, low, close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points: (H + L + C) / 3
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Support and resistance levels
    R1 = 2 * pivot - df_1d['low']
    S1 = 2 * pivot - df_1d['high']
    
    # Align 1d pivot levels to 12h timeframe (waits for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
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
            # Enter long: price above pivot AND above EMA50 (bullish alignment) + volume spike
            if (close[i] > pivot_aligned[i] and 
                close[i] > ema50[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price below pivot AND below EMA50 (bearish alignment) + volume spike
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < ema50[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot OR EMA50 turns bearish
            if (close[i] < pivot_aligned[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price crosses above pivot OR EMA50 turns bullish
            if (close[i] > pivot_aligned[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals