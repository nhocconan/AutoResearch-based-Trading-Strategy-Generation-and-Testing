#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d Williams %R extreme filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Williams %R extremes.
- Camarilla pivot levels (H3, L3) from prior 1d: Long when price > H3, Short when price < L3.
- Williams %R filter: Only trade when 1d Williams %R < -80 (oversold) for longs or > -20 (overbought) for shorts.
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying breakouts from oversold, in bear via selling breakdowns from overbought.
- Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = High + 1.1*(Low - Close)/4, L3 = Low - 1.1*(High - Close)/4
    camarilla_H3 = high_1d + 1.1 * (low_1d - close_1d) / 4
    camarilla_L3 = low_1d - 1.1 * (high_1d - close_1d) / 4
    
    # Align to 6h: use prior 1d's levels (already completed bar)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Williams %R + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade on Williams %R extremes
            if williams_r_aligned[i] < -80:  # Oversold
                if close[i] > camarilla_H3_aligned[i] and volume_spike[i]:
                    # Buy on H3 breakout from oversold
                    signals[i] = 0.25
                    position = 1
            elif williams_r_aligned[i] > -20:  # Overbought
                if close[i] < camarilla_L3_aligned[i] and volume_spike[i]:
                    # Sell on L3 breakdown from overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to Camarilla H3/L3 level or Williams %R normalizes
            if close[i] < camarilla_H3_aligned[i] or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Camarilla H3/L3 level or Williams %R normalizes
            if close[i] > camarilla_L3_aligned[i] or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dWilliamsR_Extreme_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0