# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h 1D Camarilla Pivot Breakout with Volume Confirmation
Hypothesis: Daily Camarilla pivot levels (S1, S2, R1, R2) act as key support/resistance
levels in BTC/ETH markets. Breakouts from these levels with volume confirmation
capture genuine momentum moves while avoiding false breakouts. The 12h timeframe
provides sufficient data to avoid excessive trading while capturing multi-day moves.
Volume confirmation ensures participation, and the strategy works in both bull and
bear markets by trading breakouts in either direction.
Target: 15-30 trades/year to minimize fee drag.
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
    
    # Get 1D data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's high, low, close
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), etc.
    # We'll use R1, R2, S1, S2 (more commonly traded levels)
    range_ = phigh - plow
    r1 = pclose + range_ * 1.1 / 12
    r2 = pclose + range_ * 1.1 / 6
    s1 = pclose - range_ * 1.1 / 12
    s2 = pclose - range_ * 1.1 / 6
    
    # Align to 12h timeframe (wait for daily bar to close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: 1.5x 24-period average on 12h (approx 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or 
            np.isnan(s1_12h[i]) or np.isnan(s2_12h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above R1 with volume
            if price > r1_12h[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume
            elif price < s1_12h[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns to pivot (close) or breaks S1 (reversal)
            if price < s1_12h[i] or price > r2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to pivot (close) or breaks R1 (reversal)
            if price > r1_12h[i] or price < s2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0