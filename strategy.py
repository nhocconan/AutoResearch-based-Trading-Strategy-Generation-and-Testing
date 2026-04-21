#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_VolumeSpike_Session_V1
Hypothesis: Use 1d Camarilla R1/S1 breakouts with volume confirmation and session filter (08-20 UTC) on 1h timeframe.
HTF (4h/1d) provides signal direction, 1h provides precise entry timing.
Fixed position size 0.20 to control drawdown. Target 15-35 trades/year per symbol.
Works in bull/bear via breakout logic with volume confirmation filtering low-quality signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Camarilla levels
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        hour = hours[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        in_session = (8 <= hour <= 20)  # UTC 08-20
        
        if position == 0:
            # Long: break above 1d Camarilla R1 with volume spike and in session
            if price > r1_aligned[i-1] and vol_ok and in_session:
                signals[i] = 0.20
                position = 1
            # Short: break below 1d Camarilla S1 with volume spike and in session
            elif price < s1_aligned[i-1] and vol_ok and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit on break below S1 (mean reversion) or end of session
            if price < s1_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit on break above R1 (mean reversion) or end of session
            if price > r1_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_VolumeSpike_Session_V1"
timeframe = "1h"
leverage = 1.0