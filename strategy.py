#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for volume and chop regime.
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA (1d) to avoid false breakouts.
- Choppiness regime: CHOP(14) > 61.8 = ranging (mean reversion at H3/L3), CHOP < 38.2 = trending (breakout).
- Entry: Long when price closes above H3 AND volume spike AND (CHOP > 61.8 with reversal OR CHOP < 38.2 with breakout).
         Short when price closes below L3 AND volume spike AND (CHOP > 61.8 with reversal OR CHOP < 38.2 with breakout).
- Exit: Opposite Camarilla level touch (L3 for long, H3 for short) or regime shift.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and volume/chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3) from previous day
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + (1.1 * (prev_high - prev_low) / 6)
    camarilla_l3 = prev_close - (1.1 * (prev_high - prev_low) / 6)
    
    # Align 1d Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume spike: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * volume_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = pd.Series(df_1d['high'] - df_1d['low']).abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(atr_sum + 1e-10) / np.log10(14) / np.log10(range_14 + 1e-10) * 100
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike_aligned[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                if chop_val > 61.8:  # Ranging regime: mean reversion at H3/L3
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3 and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3 and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                elif chop_val < 38.2:  # Trending regime: breakout
                    # Long when price closes above H3
                    if curr_close > h3:
                        signals[i] = 0.25
                        position = 1
                    # Short when price closes below L3
                    elif curr_close < l3:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price touches L3 OR chop shifts to strong trending
            if curr_low <= l3 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches H3 OR chop shifts to strong trending
            if curr_high >= h3 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dVolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0