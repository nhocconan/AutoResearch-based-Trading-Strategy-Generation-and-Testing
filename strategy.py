#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from daily charts act as strong support/resistance.
# Breakouts above/below key levels (H3/L3) with volume confirmation (>1.5x 20-day average)
# and aligned with 1d EMA50 trend capture sustainable moves. Designed for low trade frequency
# (<30/year) to minimize fee drift, targeting 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
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
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3 = pivot + (range_1d * 1.1 / 2)
    l3 = pivot - (range_1d * 1.1 / 2)
    h4 = pivot + (range_1d * 1.1)
    l4 = pivot - (range_1d * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or \
           np.isnan(l4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Price relative to Camarilla levels
        above_h3 = close[i] > h3_aligned[i]
        below_l3 = close[i] < l3_aligned[i]
        above_h4 = close[i] > h4_aligned[i]
        below_l4 = close[i] < l4_aligned[i]
        
        # Entry conditions
        # Long: Price breaks above H3 AND uptrend AND volume confirmation
        if above_h3 and uptrend and vol_confirm and position != 1:
            # Additional check: ensure we didn't just break above H3 in previous bar
            if i == 50 or close[i-1] <= h3_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price breaks below L3 AND downtrend AND volume confirmation
        elif below_l3 and downtrend and vol_confirm and position != -1:
            # Additional check: ensure we didn't just break below L3 in previous bar
            if i == 50 or close[i-1] >= l3_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price returns to pivot level (mean reversion)
        elif position == 1 and close[i] < pivot[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals