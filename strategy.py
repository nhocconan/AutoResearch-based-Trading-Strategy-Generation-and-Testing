#!/usr/bin/env python3
"""
Hypothesis: 6-hour price action relative to 1-day VWAP with volume confirmation.
Long when price > 1-day VWAP AND volume > 1.5x 20-period volume average.
Short when price < 1-day VWAP AND volume > 1.5x 20-period volume average.
Exit when price crosses back across 1-day VWAP.
1-day VWAP provides institutional fair value; volume confirms institutional participation.
Designed for low trade frequency by requiring both price-VWAP deviation and volume surge.
Works in bull markets (buy dips to VWAP on volume) and bear markets (sell rallies to VWAP on volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for VWAP calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pv = typical_price * df_1d['volume']
    cum_pv = pv.cumsum().values
    cum_vol = df_1d['volume'].cumsum().values
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if VWAP not ready
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above VWAP AND volume surge
            if close[i] > vwap_1d_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below VWAP AND volume surge
            elif close[i] < vwap_1d_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses back across VWAP
            if (position == 1 and close[i] < vwap_1d_aligned[i]) or \
               (position == -1 and close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_Volume_Filter"
timeframe = "6h"
leverage = 1.0