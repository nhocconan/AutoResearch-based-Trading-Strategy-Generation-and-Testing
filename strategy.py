#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Choppiness Regime
Hypothesis: On 12h timeframe, Camarilla pivot levels from 1d act as strong support/resistance.
Price approaching these levels with volume spikes indicates institutional interest.
Choppiness regime filter (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion at pivots works.
Works in bull (buy at support) and bear (sell at resistance) via mean reversion at pivot levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas (using previous day's range)
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)  # Resistance 4
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)  # Support 4
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)  # Resistance 3
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)  # Support 3
    camarilla_h2 = close_1d + (range_1d * 1.1 / 6)  # Resistance 2
    camarilla_l2 = close_1d - (range_1d * 1.1 / 6)  # Support 2
    
    # Align to 12h timeframe (shifted by 1 day for lookback)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Choppiness Index (14-period) for regime detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 > 0, 100 * np.log10(atr_sum / range_14) / np.log10(14), 50)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)  # Require significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Camarilla and volatility calculations
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price moves against position or reaches opposite pivot
        if position == 1:  # long position
            # Exit: price reaches resistance OR stops working
            if (close[i] >= h3_aligned[i] or  # Take profit at H3
                close[i] <= l4_aligned[i]):   # Stop loss if breaks below L4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches support OR stops working
            if (close[i] <= l3_aligned[i] or  # Take profit at L3
                close[i] >= h4_aligned[i]):   # Stop loss if breaks above H4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price near pivot levels with volume spike in choppy market
            # Only trade in ranging markets (Choppiness > 61.8)
            if chop[i] > 61.8:
                # Long setup: price near support levels with volume spike
                near_support = (close[i] <= l2_aligned[i] * 1.005 or  # Within 0.5% of L2
                               close[i] <= l3_aligned[i] * 1.005 or  # Within 0.5% of L3
                               close[i] <= l4_aligned[i] * 1.005)    # Within 0.5% of L4
                
                # Short setup: price near resistance levels with volume spike
                near_resistance = (close[i] >= h2_aligned[i] * 0.995 or  # Within 0.5% of H2
                                  close[i] >= h3_aligned[i] * 0.995 or  # Within 0.5% of H3
                                  close[i] >= h4_aligned[i] * 0.995)    # Within 0.5% of H4
                
                if near_support and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif near_resistance and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals