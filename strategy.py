#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts in direction of 1d ATR-filtered trend.
Camarilla levels (R1/S1) act as intraday support/resistance derived from prior 1d range.
Trend filter uses 1d ATR to confirm strong momentum (ATR > 1.5x its 20-period MA).
Volume spike (1.5x 20-bar mean) confirms breakout strength.
Avoids low-momentum breakouts that fail quickly. Designed for 4h timeframe to balance
trade frequency and fee drag, targeting ~25-40 trades/year per symbol.
Works in bull/bear regimes by requiring volume and ATR confirmation on breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, ATR trend filter, and volume context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for trend filter
    tr_1d = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    strong_trend_1d = atr_1d > (1.5 * atr_ma_1d)
    strong_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, strong_trend_1d)
    
    # Calculate Camarilla levels (R1, S1) from prior 1d range
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: 1.5x 20-period mean
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d calculations (20 for ATR MA, 20 for volume)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(strong_trend_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND strong 1d trend AND volume spike
            long_setup = (close[i] > r1_aligned[i]) and \
                         strong_trend_1d_aligned[i] and \
                         volume_spike[i]
            # Short: price breaks below S1 AND strong 1d trend AND volume spike
            short_setup = (close[i] < s1_aligned[i]) and \
                          strong_trend_1d_aligned[i] and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H3-L3 range OR trend weakens
            # Calculate H3/L3 for exit: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
            h3 = close_1d + (1.1 * camarilla_range / 6)
            l3 = close_1d - (1.1 * camarilla_range / 6)
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            in_range = (close[i] >= l3_aligned[i]) and (close[i] <= h3_aligned[i])
            if in_range or (~strong_trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H3-L3 range OR trend weakens
            h3 = close_1d + (1.1 * camarilla_range / 6)
            l3 = close_1d - (1.1 * camarilla_range / 6)
            h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
            l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
            in_range = (close[i] >= l3_aligned[i]) and (close[i] <= h3_aligned[i])
            if in_range or (~strong_trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0