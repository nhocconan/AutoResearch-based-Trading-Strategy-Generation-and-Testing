#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for regime and volume confirmation.
- Choppiness Index (CHOP) > 61.8 = ranging market (mean revert at H3/L3), CHOP < 38.2 = trending (breakout at H4/L4).
- Entry: In trending (CHOP < 38.2): Long when price breaks above H4, Short when breaks below L4.
         In ranging (CHOP > 61.8): Long when price reverses up from L3, Short when reverses down from H3.
- Volume confirmation: current 12h volume > 1.5 * 20-period 12h volume MA to filter false breakouts.
- Exit: Opposite Camarilla level touch or regime shift.
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
    
    # Get 1d data for CHOP and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # High-Low range over 14 periods
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    hl_range = highest_high - lowest_low
    
    # Chop = 100 * log10(atr_sum / hl_range) / log10(14)
    chop = 100 * np.log10(atr_sum / (hl_range + 1e-10)) / np.log10(14)
    
    # Align 1d CHOP to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels (H3, L3, H4, L4) from 1d OHLC
    # Based on previous day's close
    prev_close = pd.Series(df_1d['close']).shift(1).values
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    
    # Camarilla formula: Close + (High-Low) * multiplier
    rang = prev_high - prev_low
    h3 = prev_close + rang * 1.1 / 4
    l3 = prev_close - rang * 1.1 / 4
    h4 = prev_close + rang * 1.1 / 2
    l4 = prev_close - rang * 1.1 / 2
    
    # Align Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for CHOP and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(volume_spike[i])):
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
                if chop_val < 38.2:  # Trending regime: breakout at H4/L4
                    # Bullish breakout: price closes above H4
                    if curr_close > h4_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below L4
                    elif curr_close < l4_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                elif chop_val > 61.8:  # Ranging regime: mean reversion at H3/L3
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price touches L3 (mean reversion) or regime shifts to trending
            if curr_low <= l3_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches H3 (mean reversion) or regime shifts to trending
            if curr_high >= h3_aligned[i] or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dChopRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0