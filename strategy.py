#!/usr/bin/env python3
"""
6h_HTF_Structure_Aligned_Trend
Hypothesis: Uses weekly structure (higher highs/lows) as trend filter and daily price action for entry timing on 6h timeframe.
Designed to capture medium-term trends while avoiding counter-trend trades in choppy markets. 
Weekly structure provides robust trend identification, while daily price action offers precise entries.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Should work in both bull (follow structure) and bear (respect structure reversals) markets.
"""

name = "6h_HTF_Structure_Aligned_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly Structure: Higher Highs/Higher Lows (uptrend) or Lower Lows/Lower Highs (downtrend) ---
    # Calculate weekly swing points
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    
    # Weekly pivot highs and lows (3-bar structure)
    whigh_idx = np.argmax(whigh) if len(whigh) >= 3 else 0
    wlow_idx = np.argmin(wlow) if len(wlow) >= 3 else 0
    
    # Simplified: use weekly close vs previous weekly close for trend
    wclose = df_1w['close'].values
    wtrend = np.where(wclose >= np.roll(wclose, 1), 1, -1)  # 1=uptrend, -1=downtrend
    wtrend[0] = 1  # Initialize
    
    # Align weekly trend to 6h
    wtrend_aligned = align_htf_to_ltf(prices, df_1w, wtrend.astype(float))
    
    # --- Daily Price Action: Engulfing candles for entry ---
    do = df_1d['open'].values
    dh = df_1d['high'].values
    dl = df_1d['low'].values
    dc = df_1d['close'].values
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bull_eng = (dc > do) & (dc[:-1] < do[:-1]) & (dc >= do[:-1]) & (do <= dc[:-1])
    # Bearish engulfing: current red candle engulfs previous green candle
    bear_eng = (dc < do) & (dc[:-1] > do[:-1]) & (dc <= do[:-1]) & (do >= dc[:-1])
    
    # Pad arrays to match daily length
    bull_eng = np.concatenate([[False], bull_eng])
    bear_eng = np.concatenate([[False], bear_eng])
    
    # Align engulfing signals to 6h
    bull_eng_aligned = align_htf_to_ltf(prices, df_1d, bull_eng.astype(float))
    bear_eng_aligned = align_htf_to_ltf(prices, df_1d, bear_eng.astype(float))
    
    # --- Volume confirmation on 6h ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wtrend_aligned[i]) or 
            np.isnan(bull_eng_aligned[i]) or
            np.isnan(bear_eng_aligned[i]) or
            np.isnan(vol_ma.values[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry logic: follow weekly structure with daily price action confirmation
        if position == 0:
            # Long: weekly uptrend + daily bullish engulfing + volume spike
            if (wtrend_aligned[i] > 0 and 
                bull_eng_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + daily bearish engulfing + volume spike
            elif (wtrend_aligned[i] < 0 and 
                  bear_eng_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit: weekly structure reversal
            if position == 1 and wtrend_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and wtrend_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals