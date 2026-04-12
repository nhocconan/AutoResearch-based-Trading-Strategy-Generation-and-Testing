#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v2
# Uses daily Camarilla pivot levels (H3/L3) with volume confirmation and volatility filter.
# Trades breakouts of H3/L3 levels in trending markets (ATR-based volatility filter).
# Designed to work in both bull and bear markets by capturing momentum breakouts.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_1d_camarilla_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day (H3/L3 for tighter stops)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas - using H3/L3 for tighter, more frequent signals
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    
    # Align to 4h timeframe
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: volume > 1.3 * 20-period average (slightly relaxed)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Volatility filter: only trade when volatility is above average (avoid low vol chop)
    # Calculate ATR(14) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr > atr_ma  # Trade when volatility is above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h3_level[i]) or np.isnan(l3_level[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and volatility filters
        if not (vol_confirm[i] and vol_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H3 with volume
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3 with volume
        elif close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l3_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h3_level[i] and position == -1:
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