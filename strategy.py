#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla levels from 1d: H3 (resistance) and L3 (support) act as magnet levels
- Long when price breaks above H3 with volume > 1.5 * volume_ma20 AND close > 1d EMA34 (uptrend)
- Short when price breaks below L3 with volume > 1.5 * volume_ma20 AND close < 1d EMA34 (downtrend)
- Exit on opposite Camarilla level touch (H4 for longs, L4 for shorts) or EMA trend reversal
- Designed to capture institutional order flow around key daily levels with trend alignment
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 80-120 total trades over 4 years (20-30/year) - within sustainable range
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
    
    # Calculate 1d HTF data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Camarilla pivot levels from previous 1d bar
    # H4 = close + 1.1 * (high - low) / 2
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # L4 = close - 1.1 * (high - low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (using previous bar to avoid look-ahead)
    camarilla_high = np.roll(high_1d, 1)
    camarilla_low = np.roll(low_1d, 1)
    camarilla_close = np.roll(close_1d, 1)
    camarilla_high[0] = np.nan  # First bar has no previous
    camarilla_low[0] = np.nan
    camarilla_close[0] = np.nan
    
    camarilla_range = camarilla_high - camarilla_low
    h3 = camarilla_close + 1.1 * camarilla_range / 4
    l3 = camarilla_close - 1.1 * camarilla_range / 4
    h4 = camarilla_close + 1.1 * camarilla_range / 2
    l4 = camarilla_close - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: volume > 1.5 * 20-period volume MA
    # Calculate on 6h data
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need volume MA20 and EMA34 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with volume spike AND uptrend
            if close[i] > h3_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike AND downtrend
            elif close[i] < l3_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches H4 (profit target) OR EMA trend reverses
            if close[i] >= h4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches L4 (profit target) OR EMA trend reverses
            if close[i] <= l4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0