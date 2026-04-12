#!/usr/bin/env python3
"""
4h_1d_camarilla_reversal_v1
Hypothesis: 4-hour strategy using daily Camarilla pivot levels for mean-reversion entries in both bull and bear markets.
Goes long at L3 support and short at H3 resistance with volume confirmation and volatility filter.
Designed for low trade frequency (<40/year) to minimize fee drag and work in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L3 = C - (H - L) * 1.1 / 2
    # H3 = C + (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    L3_1d = close_1d - (range_1d * 1.1 / 2.0)
    H3_1d = close_1d + (range_1d * 1.1 / 2.0)
    
    # Align to 4h timeframe
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    
    # Calculate 4-period RSI for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(L3_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        # Long at L3 support with oversold RSI
        if vol_ok and low[i] <= L3_1d_aligned[i] * 1.002 and rsi[i] < 30:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        # Short at H3 resistance with overbought RSI
        elif vol_ok and high[i] >= H3_1d_aligned[i] * 0.998 and rsi[i] > 70:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        # Exit on middle of range (pivot) or RSI normalization
        elif position == 1 and (close[i] >= pivot_1d[i] * 0.998 or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= pivot_1d[i] * 1.002 or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_reversal_v1"
timeframe = "4h"
leverage = 1.0