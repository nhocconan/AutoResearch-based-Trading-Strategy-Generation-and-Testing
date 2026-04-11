#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot levels with 1d volume confirmation and ATR stoploss
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance zones. Price retracements to these levels with volume confirmation offer high-probability reversal entries. The 1d timeframe provides volume context to distinguish institutional participation from noise. Designed for low trade frequency (~15-30/year) to minimize fee drift. Works in both bull and bear markets by fading extremes at statistically significant levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
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
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    # Camarilla levels based on previous day's range
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate Camarilla levels for current day based on previous day's OHLC
    # Range = previous day high - previous day low
    # Levels: Close ± (Range * multiplier / 11)
    # Key levels: L3, L4, L5, L6 (support) and H3, H4, H5, H6 (resistance)
    # We focus on L3, L4, H3, H4 as primary levels
    range_1d = prev_high - prev_low
    camarilla_l3 = prev_close - (range_1d * 1.1 / 12)  # ~0.0916 * range
    camarilla_l4 = prev_close - (range_1d * 1.1 / 6)   # ~0.1833 * range
    camarilla_h3 = prev_close + (range_1d * 1.1 / 12)  # ~0.0916 * range
    camarilla_h4 = prev_close + (range_1d * 1.1 / 6)   # ~0.1833 * range
    
    # Align Camarilla levels to 12h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
    # 1d volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # ATR for stoploss (using 12h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        # Note: We use 12h volume from prices, compare to aligned 1d volume average
        vol_confirm = volume[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Entry conditions
        # Long: Price retraces to L3/L4 support with volume confirmation
        long_signal = ((low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]) and
                       vol_confirm and position != 1)
        
        # Short: Price retraces to H3/H4 resistance with volume confirmation
        short_signal = ((high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]) and
                        vol_confirm and position != -1)
        
        # Stoploss: ATR-based (2.5 * ATR from entry)
        # We track entry price implicitly through position and use current bar's extreme
        if position == 1 and low[i] < camarilla_l4_aligned[i] - 2.5 * atr[i]:
            # Stoploss hit for long
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and high[i] > camarilla_h4_aligned[i] + 2.5 * atr[i]:
            # Stoploss hit for short
            signals[i] = 0.0
            position = 0
            continue
        
        # Execute signals
        if long_signal:
            position = 1
            signals[i] = 0.25
        elif short_signal:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals