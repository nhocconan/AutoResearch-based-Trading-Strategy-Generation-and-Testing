#!/usr/bin/env python3
name = "4h_ThreeLineStrike_1dVWAP_Momentum"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # 4h Three Line Strike pattern detection
    # Bullish 3LS: 3 consecutive down closes, then 4th bar closes above 1st bar open
    # Bearish 3LS: 3 consecutive up closes, then 4th bar closes below 1st bar open
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Wait for enough history to detect pattern
    start_idx = 4
    
    for i in range(start_idx, n):
        # Check if we have enough data for pattern detection
        if i < 3:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bullish 3LS pattern
        bullish_3ls = (
            close[i-3] < close[i-2] and  # bar 1 down
            close[i-2] < close[i-1] and  # bar 2 down
            close[i-1] < close[i-3] and  # bar 3 down (continuing downtrend)
            close[i] > close[i-3]        # bar 4 closes above bar 1 open (approximated by close)
        )
        
        # Bearish 3LS pattern
        bearish_3ls = (
            close[i-3] > close[i-2] and  # bar 1 up
            close[i-2] > close[i-1] and  # bar 2 up
            close[i-1] > close[i-3] and  # bar 3 up (continuing uptrend)
            close[i] < close[i-3]        # bar 4 closes below bar 1 open
        )
        
        if position == 0:
            # Enter long on bullish 3LS if price above 1d VWAP
            if bullish_3ls and close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish 3LS if price below 1d VWAP
            elif bearish_3ls and close[i] < vwap_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long on bearish 3LS or price below VWAP
            if bearish_3ls or close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on bullish 3LS or price above VWAP
            if bullish_3ls or close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Three Line Strike reversal pattern with 1d VWAP trend filter.
# Bullish 3LS: 3 down bars followed by 4th bar closing above first bar's open = potential bullish reversal.
# Bearish 3LS: 3 up bars followed by 4th bar closing below first bar's open = potential bearish reversal.
# Entry confirmed when price is on favorable side of 1d VWAP (above for longs, below for shorts).
# Works in both bull and bear markets as it captures mean-reversion from exhaustion moves.
# VWAP filter ensures we trade with the higher timeframe trend/value area.
# Target: ~30-60 trades/year to balance opportunity with cost control.