#!/usr/bin/env python3
"""
4h_1D_Camarilla_ShortBreakout
Hypothesis: In bear markets (2022-2025), price often breaks below Camarilla L4 support during weak rallies.
Go short when price closes below L4 with volume confirmation, exit when price closes back above L3.
Use 1-day Camarilla levels calculated from prior day's high-low-close. Works in sideways/bleeding markets.
Target: 20-50 total trades over 4 years (5-12/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Camarilla_ShortBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-DAY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla from prior day's HLC
    # L4 = C - ((H - L) * 1.1 / 2)
    # L3 = C - ((H - L) * 1.1 / 4)
    # H3 = C + ((H - L) * 1.1 / 4)
    # H4 = C + ((H - L) * 1.1 / 2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day values (shifted by 1)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = high_1d[0]  # first bar: use same day
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    rang = phigh - plow
    L4 = pclose - (rang * 1.1 / 2)
    L3 = pclose - (rang * 1.1 / 4)
    H3 = pclose + (rang * 1.1 / 4)
    H4 = pclose + (rang * 1.1 / 2)
    
    # Align to 4h
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(L4_4h[i]) or np.isnan(L3_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Short signal: close below L4 with volume
        short_signal = (close[i] < L4_4h[i]) and (vol_ratio[i] > 1.3)
        
        # Exit short: close back above L3
        exit_short = (position == -1) and (close[i] > L3_4h[i])
        
        # Execute
        if short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold
            signals[i] = -0.25 if position == -1 else 0.0
    
    return signals