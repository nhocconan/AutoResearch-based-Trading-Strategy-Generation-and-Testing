#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filter
Hypothesis: Uses Camarilla pivot levels from 1d timeframe with breakout logic on 12h.
Enters long when price breaks above R1 with 1d uptrend and volume spike,
short when breaks below S1 with 1d downtrend and volume spike.
Designed for 12-37 trades/year with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate previous day's Camarilla levels
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = typical_price + range_hl * 1.1 / 12
    s1 = typical_price - range_hl * 1.1 / 12
    
    # Align to 12h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1d trend: EMA25 > EMA50 for uptrend, EMA25 < EMA50 for downtrend
    ema25 = pd.Series(df_1d['close']).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema25_aligned = align_htf_to_ltf(prices, df_1d, ema25)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # 12h volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema25_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with 1d uptrend and volume spike
            if close[i] > r1_aligned[i] and ema25_aligned[i] > ema50_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1d downtrend and volume spike
            elif close[i] < s1_aligned[i] and ema25_aligned[i] < ema50_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or trend weakens
            if close[i] < s1_aligned[i] or ema25_aligned[i] <= ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or trend weakens
            if close[i] > r1_aligned[i] or ema25_aligned[i] >= ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filter"
timeframe = "12h"
leverage = 1.0