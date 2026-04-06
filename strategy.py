#!/usr/bin/env python3
"""
12h Camarilla pivot with 1d volume spike and choppiness regime filter.
Hypothesis: Price reversions at Camarilla levels (H3/L3) during high-volume,
non-trending markets capture mean-reversion swings in both bull and bear regimes.
Choppiness filter avoids trending markets where mean reversion fails.
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14265_12h_camarilla_pivot_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index (high = ranging, low = trending)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.fillna(50).values  # neutral when undefined

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    # H3/L3 = close +- 1.1*(high-low)/6
    camarilla_h3 = close_1d + (1.1 * (high_1d - low_1d) / 6)
    camarilla_l3 = close_1d - (1.1 * (high_1d - low_1d) / 6)
    
    # Align to 12h timeframe (shifted by 1 for completed day only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 2x 20-period average (spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    # Choppiness filter: > 61.8 = ranging (good for mean reversion)
    chop = calculate_choppiness(high, low, close, 14)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup (max of 20 for volume, 14 for chop)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Mean reversion at Camarilla H3/L3 with volume and chop filter
        # Long: near L3 (< 0.2% above) + volume spike + choppy market
        # Short: near H3 (< 0.2% below) + volume spike + choppy market
        near_l3 = close[i] <= camarilla_l3_aligned[i] * 1.002
        near_h3 = close[i] >= camarilla_h3_aligned[i] * 0.998
        
        if position == 0:
            if near_l3 and vol_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif near_h3 and vol_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price reaches midpoint or opposite level
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if close[i] >= midpoint or close[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price reaches midpoint or opposite level
            midpoint = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
            if close[i] <= midpoint or close[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals