#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d choppiness regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for regime and volume filters.
- Choppiness Index (CHOP) > 61.8 = ranging market (mean reversion at H3/L3), CHOP < 38.2 = trending (breakout).
- Entry: In trending (CHOP < 38.2): Long on break above H3, Short on break below L3.
         In ranging (CHOP > 61.8): Long on reversal from L3 (close > low after touch), Short on reversal from H3 (close < high after touch).
- Volume confirmation: current volume > 1.5 * 24-period volume MA (to avoid false signals).
- Exit: Opposite Camarilla level breakout or regime shift.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for CHOP and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = 100 * np.log10(atr / (hh - ll + 1e-10)) / np.log10(14)
    
    # Align 1d CHOP to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d candle
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    tp_shifted = typical_price.shift(1)  # Use previous day's typical price
    range_ = df_1d['high'] - df_1d['low']
    range_shifted = range_.shift(1)
    
    # Camarilla H3 = TP + (H-L) * 1.1/4
    # Camarilla L3 = TP - (H-L) * 1.1/4
    h3 = tp_shifted + range_shifted * 1.1 / 4.0
    l3 = tp_shifted - range_shifted * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    
    # Volume confirmation: current volume > 1.5 * 24-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 24)  # Need enough 1d bars for CHOP/Camarilla and 12h bars for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if chop_val < 38.2:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above H3
                    if curr_close > h3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below L3
                    elif curr_close < l3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at extremes
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
                # In between 38.2 and 61.8: neutral/choppy, no entries
        elif position == 1:
            # Long exit: price closes below L3 OR regime shifts to neutral/choppy
            if curr_close < l3_aligned[i] or (chop_val >= 38.2 and chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR regime shifts to neutral/choppy
            if curr_close > h3_aligned[i] or (chop_val >= 38.2 and chop_val <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_CamarillaH3L3_1dChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0