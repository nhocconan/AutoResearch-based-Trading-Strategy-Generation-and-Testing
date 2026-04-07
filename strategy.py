#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: Camarilla pivot levels from 1d timeframe provide strong support/resistance. 
Long when price breaks above R4 with volume confirmation, short when price breaks below S4 with volume confirmation. 
Use 1w EMA40 as trend filter to avoid counter-trend trades. Works in both bull and bear by following higher timeframe trend.
Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's high, low, close
    # R4 = C + (H-L) * 1.1/2, S4 = C - (H-L) * 1.1/2
    # where C = (H+L+C)/3 (typical price)
    # Actually standard Camarilla uses previous day's H,L,C:
    # R4 = CLOSE + (HIGH - LOW) * 1.1/2
    # S4 = CLOSE - (HIGH - LOW) * 1.1/2
    # But we'll use typical price as pivot for simplicity
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Pivot point (typical price)
    p = (phigh + plow + pclose) / 3.0
    # Camarilla levels
    r4 = p + (phigh - plow) * 1.1 / 2.0
    s4 = p - (phigh - plow) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1w EMA40 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_40 = df_1w['close'].ewm(span=40, adjust=False).mean()
    ema_40_aligned = align_htf_to_ltf(prices, df_1w, ema_40.values)
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_40_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price drops below S4 or closes below EMA40
            if close[i] < s4_aligned[i] or close[i] < ema_40_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price rises above R4 or closes above EMA40
            if close[i] > r4_aligned[i] or close[i] > ema_40_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above R4 with volume and above EMA40 (uptrend)
            if (close[i] > r4_aligned[i] and vol_confirm and 
                close[i] > ema_40_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below S4 with volume and below EMA40 (downtrend)
            elif (close[i] < s4_aligned[i] and vol_confirm and 
                  close[i] < ema_40_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals