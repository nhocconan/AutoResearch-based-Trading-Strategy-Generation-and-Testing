#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_v1
# Strategy: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from 1d act as key support/resistance. Breaking above/below
# these levels with volume confirmation indicates institutional interest. The 1d EMA50 filter
# ensures trades align with higher timeframe trend, reducing whipsaws in sideways markets.
# Designed for low trade frequency (~15-30/year) to minimize fee drag while capturing
# significant moves in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for 1d (used in Camarilla calculation)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[0], tr])  # align with same length
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.1 * (high - low) / 2
    # L4 = close - 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # H2 = close + 1.1 * (high - low) / 6
    # L2 = close - 1.1 * (high - low) / 6
    # H1 = close + 1.1 * (high - low) / 12
    # L1 = close - 1.1 * (high - low) / 12
    
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * range_1d / 2
    camarilla_l4 = close_1d - 1.1 * range_1d / 2
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    camarilla_h2 = close_1d + 1.1 * range_1d / 6
    camarilla_l2 = close_1d - 1.1 * range_1d / 6
    camarilla_h1 = close_1d + 1.1 * range_1d / 12
    camarilla_l1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout conditions
        breakout_above_h4 = close[i] > h4_aligned[i]
        breakdown_below_l4 = close[i] < l4_aligned[i]
        
        # Trend filter: align with 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: break above H4 AND uptrend AND volume confirmation
        if breakout_above_h4 and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: break below L4 AND downtrend AND volume confirmation
        elif breakdown_below_l4 and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: break of opposite level (H4 for shorts, L4 for longs) or trend change
        elif position == 1 and (breakdown_below_l4 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_above_h4 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals