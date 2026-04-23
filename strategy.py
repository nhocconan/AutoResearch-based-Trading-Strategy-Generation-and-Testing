#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long: Close breaks above Camarilla R1 + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period avg
- Short: Close breaks below Camarilla S1 + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period avg
- Exit: Close crosses Camarilla H3/L3 levels (mean reversion to pivot center)
- Uses Camarilla pivot levels from daily HTF for structure, volume confirmation to avoid false breakouts
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
- Works in bull markets (breakouts with trend) and bear markets (mean reversion at pivots)
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
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels from 1d HTF data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_range = high_1d - low_1d
    camarilla_h5 = close_1d + camarilla_range * 1.1 / 2
    camarilla_h4 = close_1d + camarilla_range * 1.1 / 4
    camarilla_h3 = close_1d + camarilla_range * 1.1 / 6
    camarilla_l3 = close_1d - camarilla_range * 1.1 / 6
    camarilla_l4 = close_1d - camarilla_range * 1.1 / 4
    camarilla_l5 = close_1d - camarilla_range * 1.1 / 2
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_r1 = camarilla_h4  # R1 = H4
    camarilla_s1 = camarilla_l4  # S1 = L4
    camarilla_r1_aligned = camarilla_h4_aligned
    camarilla_s1_aligned = camarilla_l4_aligned
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Need 20 for volume MA, 1 for HTF data (already aligned)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above Camarilla R1 + price > 1d EMA34 (uptrend) + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S1 + price < 1d EMA34 (downtrend) + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below Camarilla H3 (mean reversion)
            if close[i] < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above Camarilla L3 (mean reversion)
            if close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0