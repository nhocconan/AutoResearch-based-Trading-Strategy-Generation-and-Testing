#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot breakout with daily volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels derived from daily price action act as
# institutional support/resistance. Breakouts with volume confirmation capture
# institutional flow. Works in bull (breakouts) and bear (mean reversion at H3/L3).
# Target: 20-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # H3 = C + 1.0*(H-L), L3 = C - 1.0*(H-L)
    # H2 = C + 0.5*(H-L), L2 = C - 0.5*(H-L)
    # H1 = C + 0.25*(H-L), L1 = C - 0.25*(H-L)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data
    H_L = prev_high - prev_low
    H4 = prev_close + 1.5 * H_L
    L4 = prev_close - 1.5 * H_L
    H3 = prev_close + 1.0 * H_L
    L3 = prev_close - 1.0 * H_L
    H2 = prev_close + 0.5 * H_L
    L2 = prev_close - 0.5 * H_L
    H1 = prev_close + 0.25 * H_L
    L1 = prev_close - 0.25 * H_L
    
    # Align to 4h timeframe (use previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long breakout above H4 with volume
        if close[i] > H4_aligned[i] and volume_spike[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short breakdown below L4 with volume
        elif close[i] < L4_aligned[i] and volume_spike[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Mean reversion: long at L3, short at H3
        elif close[i] < L3_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] > H3_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Camarilla level touch
        elif position == 1 and close[i] > H3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < L3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals