#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_1w_camarilla_confluence
# Use 1d and 1w Camarilla levels with confluence: both H4 levels must agree for long,
# both L4 levels for short. Add volume confirmation and chop regime filter to avoid false signals.
# Timeframe: 12h - balances trade frequency and signal quality. Works in bull (breakouts above H4)
# and bear (breakdowns below L4) by using higher timeframe structure. Target: 15-30 trades/year.

name = "12h_1d_1w_camarilla_confluence"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for confluence
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla from previous day
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    range_1d = high_1d_prev - low_1d_prev
    h4_1d = close_1d_prev + range_1d * 1.1 / 2
    l4_1d = close_1d_prev - range_1d * 1.1 / 2
    
    # Calculate 1w Camarilla from previous week
    high_1w_prev = df_1w['high'].shift(1).values
    low_1w_prev = df_1w['low'].shift(1).values
    close_1w_prev = df_1w['close'].shift(1).values
    range_1w = high_1w_prev - low_1w_prev
    h4_1w = close_1w_prev + range_1w * 1.1 / 2
    l4_1w = close_1w_prev - range_1w * 1.1 / 2
    
    # Align to 12h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h4_1w_aligned = align_htf_to_ltf(prices, df_1w, h4_1w)
    l4_1w_aligned = align_htf_to_ltf(prices, df_1w, l4_1w)
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # Chop regime filter: avoid choppy markets (CHOP > 61.8)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((highest_high - lowest_low) / (atr * np.sqrt(14))) / np.log10(14)
    chop_filter = chop < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(h4_1w_aligned[i]) or np.isnan(l4_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and chop filters
        if not (vol_confirm[i] and chop_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above BOTH 1d H4 AND 1w H4 with volume
        if close[i] > h4_1d_aligned[i] and close[i] > h4_1w_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below BOTH 1d L4 AND 1w L4 with volume
        elif close[i] < l4_1d_aligned[i] and close[i] < l4_1w_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price breaks below 1d L4 (for long) or above 1d H4 (for short)
        elif close[i] < l4_1d_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h4_1d_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals