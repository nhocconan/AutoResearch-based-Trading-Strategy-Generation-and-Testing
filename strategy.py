#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v28
# Strategy: 4h price breaking above/below Camarilla pivot levels (from 1d) with volume confirmation and volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts with volume indicate institutional participation.
# Volatility filter (ATR-based) avoids whipsaws in low-volatility environments. Works in bull markets via breakout continuation
# and bear markets via breakdown continuation. Designed for low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v28"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # Formula: Close + (High - Low) * multiplier / 11
    # We use H3, L3 for breakout/breakdown
    rng = high_1d - low_1d
    camarilla_h3 = close_1d + rng * 1.1 / 11
    camarilla_l3 = close_1d - rng * 1.1 / 11
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    # Volatility filter: ATR(14) > 20-period ATR average (avoid low-volatility whipsaws)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(vol_confirm[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions
        # Long: price breaks above H3 with volume and volatility confirmation
        if breakout_up and vol_confirm[i] and vol_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price breaks below L3 with volume and volatility confirmation
        elif breakout_down and vol_confirm[i] and vol_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to the opposite Camarilla level (mean reversion)
        elif position == 1 and close[i] < camarilla_l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > camarilla_h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals