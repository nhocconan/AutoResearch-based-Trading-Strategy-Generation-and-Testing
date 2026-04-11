#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v26
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels provide high-probability reversal/breakout levels.
# Long: Break above H4 with 1d volume > 1.5x 20-day avg and chop < 61.8.
# Short: Break below L4 with same conditions.
# Uses daily timeframe for volume/chop to reduce noise and improve reliability.
# Designed for low trade frequency (~20-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v26"
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
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d volume and chop filter
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Choppiness Index (14-period)
    atr_14 = []
    for i in range(len(df_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = np.max([
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            ])
            atr_14.append(tr)
    atr_14 = np.array(atr_14)
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        vol_1d_current = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]
        vol_confirm = vol_1d_current > 1.5 * vol_avg_20_1d[i]
        
        # Chop regime: chop < 61.8 for trending market
        chop_filter = chop_aligned[i] < 61.8
        
        # Breakout conditions
        breakout_up = high[i] > camarilla_h4_aligned[i]
        breakout_down = low[i] < camarilla_l4_aligned[i]
        
        # Entry conditions
        # Long: Break above H4 with volume confirmation and trending chop
        if breakout_up and vol_confirm and chop_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Break below L4 with volume confirmation and trending chop
        elif breakout_down and vol_confirm and chop_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (reversion to mean)
        elif position == 1 and low[i] < camarilla_l4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > camarilla_h4_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals