# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_PriceAction_SupportResistance_Breakout
Hypothesis: Trade breakouts at key support/resistance levels (prior day high/low) with 4h trend filter and volume confirmation.
Works in bull/bear markets by aligning with higher timeframe trend. Uses price action rather than calculated pivots
to avoid curve-fitting. Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_PriceAction_SupportResistance_Breakout"
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
    
    # === 4h Trend Filter (EMA 34) ===
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Daily Data for Support/Resistance (prior day high/low) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high and low for support/resistance
    ph_1d = df_1d['high'].values
    pl_1d = df_1d['low'].values
    
    # Align to 4h timeframe (prior day levels available after day close)
    ph_4h = align_htf_to_ltf(prices, df_1d, ph_1d)
    pl_4h = align_htf_to_ltf(prices, df_1d, pl_1d)
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and daily data)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ph_4h[i]) or np.isnan(pl_4h[i]) or 
            np.isnan(ema34[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above prior day high with uptrend and volume
            if (close[i] > ph_4h[i] and 
                close[i] > ema34[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below prior day low with downtrend and volume
            elif (close[i] < pl_4h[i] and 
                  close[i] < ema34[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below prior day low (mean reversion)
            if close[i] < pl_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above prior day high (mean reversion)
            if close[i] > ph_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals