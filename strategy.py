#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volatility filter and volume spike confirmation.
- Uses 1d Camarilla pivots for structure and 1d ATR(14) to filter low-volatility breakouts
- Volume spike >2.0x average to confirm institutional participation
- Position size: 0.25 discrete level to minimize fee churn
- Trend-neutral: works in any market regime via volatility filter
- Designed for 12-37 trades/year on 12h timeframe to avoid fee drag
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pp_1d + range_1d * 1.1 / 4
    l3_1d = pp_1d - range_1d * 1.1 / 4
    h4_1d = pp_1d + range_1d * 1.1 / 2
    l4_1d = pp_1d - range_1d * 1.1 / 2
    
    # 1d ATR(14) for volatility filter
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or
            np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: only trade when ATR > 0.5 * 20-period ATR average (avoid low-vol chop)
        atr_ma = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr_14_1d_aligned[i] > 0.5 * atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up_h3 = close[i] > h3_1d_aligned[i-1]  # Close above prior 1d H3
        breakout_down_l3 = close[i] < l3_1d_aligned[i-1]  # Close below prior 1d L3
        
        if position == 0:
            # Long: 1d Camarilla H3 breakout up AND volume confirmation AND volatility filter
            if breakout_up_h3 and volume_confirm and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: 1d Camarilla L3 breakout down AND volume confirmation AND volatility filter
            elif breakout_down_l3 and volume_confirm and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla L4 break down
            if close[i] < l4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Camarilla H4 break up
            if close[i] > h4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0