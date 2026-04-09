#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_chop_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter.
# In choppy markets (choppiness index > 61.8): mean revert at H3/L3 levels (long at L3, short at H3).
# In trending markets (choppiness index < 38.2): breakout continuation (long above H4, short below L4).
# Uses volume spike (>1.5x 20-period average) to confirm momentum and reduce false signals.
# Designed for low turnover (target: 50-150 total trades over 4 years) by requiring regime alignment and volume confirmation.
# Works in both bull and bear markets via regime adaptation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the given period"""
    # Typical price
    typical_price = (high + low + close) / 3
    # Range
    range_ = high - low
    
    # Camarilla levels
    h4 = close + range_ * 1.1 / 2
    h3 = close + range_ * 1.1 / 4
    l3 = close - range_ * 1.1 / 4
    l4 = close - range_ * 1.1 / 2
    
    return h4, h3, l3, l4

def calculate_choppiness(high, low, close, window=14):
    """Calculate Choppiness Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over window
    atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
    
    # Highest high and lowest low over window
    hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
    ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
    # Handle division by zero and invalid values
    chop = np.where((hh - ll) > 0, chop, 50.0)
    return chop

name = "12h_camarilla_pivot_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d HTF Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    h1 = df_1d['high'].values
    l1 = df_1d['low'].values
    c1 = df_1d['close'].values
    
    h4_1d, h3_1d, l3_1d, l4_1d = calculate_camarilla(h1, l1, c1)
    
    # Align HTF Camarilla to LTF (12h)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # 12h LTF Choppiness Index for regime detection (14-period)
    chop = calculate_choppiness(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(h4_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop[i] > 61.8:  # Choppy regime: mean revert at L3
                if close[i] > l3_1d_aligned[i]:  # Exit long when price moves above L3
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif chop[i] < 38.2:  # Trending regime: trail with L4
                if close[i] < l4_1d_aligned[i]:  # Exit long when price breaks below L4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Transition regime: exit on opposite signal
                if close[i] < h3_1d_aligned[i]:  # Exit on short signal
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop[i] > 61.8:  # Choppy regime: mean revert at H3
                if close[i] < h3_1d_aligned[i]:  # Exit short when price moves below H3
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif chop[i] < 38.2:  # Trending regime: trail with H4
                if close[i] > h4_1d_aligned[i]:  # Exit short when price breaks above H4
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Transition regime: exit on opposite signal
                if close[i] > l3_1d_aligned[i]:  # Exit on long signal
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                if chop[i] > 61.8:  # Choppy regime: mean reversion
                    # Enter long at L3 support
                    if close[i] <= l3_1d_aligned[i] * 1.001:  # Small buffer for entry
                        position = 1
                        signals[i] = 0.25
                    # Enter short at H3 resistance
                    elif close[i] >= h3_1d_aligned[i] * 0.999:  # Small buffer for entry
                        position = -1
                        signals[i] = -0.25
                elif chop[i] < 38.2:  # Trending regime: breakout
                    # Enter long on breakout above H4
                    if close[i] > h4_1d_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Enter short on breakdown below L4
                    elif close[i] < l4_1d_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                # In transition regime (38.2 <= chop <= 61.8): wait for clearer signal
    
    return signals