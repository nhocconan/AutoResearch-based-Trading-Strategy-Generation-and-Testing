#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and choppiness regime filter.
- Uses Camarilla pivot levels (H3, L3) from prior completed 1d candles for stronger breakouts.
- Volume confirmation: current volume > 2.0x 20-bar average to ensure momentum.
- Regime filter: Choppiness Index(14) < 38.2 to ensure trending market (avoid sideways chop).
- Designed for 4h timeframe to capture strong trending moves in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivots and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from prior completed 1d candles
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla width
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # H3 and L3 levels
    h3_1d = close_1d + camarilla_width * 2.618  # H3 = close + width * 2.618
    l3_1d = close_1d - camarilla_width * 2.618  # L3 = close - width * 2.618
    
    # Align H3 and L3 to 4h timeframe (wait for 1d bar to close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d Choppiness Index regime filter (trending when < 38.2)
    # True range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Choppiness Index
    chop_1d = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    # Align chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        # Regime filter: trending market (chop < 38.2)
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: breakout above H3 AND volume confirmation AND trending regime
            if close[i] > h3_1d_aligned[i] and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L3 AND volume confirmation AND trending regime
            elif close[i] < l3_1d_aligned[i] and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below L3 OR loss of volume confirmation OR choppy regime
            if (close[i] < l3_1d_aligned[i] or not volume_confirm or not trending_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above H3 OR loss of volume confirmation OR choppy regime
            if (close[i] > h3_1d_aligned[i] or not volume_confirm or not trending_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3_L3_Breakout_1dVolSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0