#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversion with Volume and Volume Spike
Hypothesis: 12h price reversals at Camarilla pivot levels (calculated from 1d) capture mean reversion.
Volume spike confirms institutional interest at pivot levels. Works in bull (buy dips) and bear (sell rallies).
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4 (most significant for reversals)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (previous day's levels)
    camarilla_h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 2.0 * 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    volume_spike = volume > (vol_ema * 2.0)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For ATR and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h4_12h[i]) or np.isnan(camarilla_l4_12h[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or price reaches Camarilla H4 (take profit)
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] >= camarilla_h4_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or price reaches Camarilla L4 (take profit)
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] <= camarilla_l4_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at Camarilla L4/H4 with volume spike
            # Long when price touches L4 and bounces up
            # Short when price touches H4 and bounces down
            tol = 0.001 * close[i]  # 0.1% tolerance for level touch
            
            touch_l4 = (abs(low[i] - camarilla_l4_12h[i]) <= tol) and (close[i] > camarilla_l4_12h[i])
            touch_h4 = (abs(high[i] - camarilla_h4_12h[i]) <= tol) and (close[i] < camarilla_h4_12h[i])
            
            long_entry = touch_l4 and volume_spike[i]
            short_entry = touch_h4 and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals