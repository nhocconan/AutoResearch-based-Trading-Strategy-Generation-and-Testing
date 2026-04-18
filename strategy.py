#!/usr/bin/env python3
"""
12h_Ichimoku_Kijun_Breakout_With_Volume
Hypothesis: Ichimoku Kijun-sen (base line) breakouts on 12h with volume confirmation.
Long when price crosses above Kijun with volume spike; short when crosses below with volume spike.
Ichimoku works in both bull/bear markets by capturing momentum shifts. Volume filter reduces false breakouts.
Designed for low trade frequency (12-37/year) on 12h timeframe to avoid fee drag.
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
    
    # Ichimoku components (26-period)
    # Tenkan-sen (conversion line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (base line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (leading span A): (Tenkan + Kijun)/2 shifted 26 periods
    # Not needed for breakout logic
    
    # Senkou Span B (leading span B): (52-period high + 52-period low)/2 shifted 52 periods
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Chikou Span (lagging span): close shifted -22 periods
    # Not needed for breakout logic
    
    # Volume spike: >1.8x 20-period average (moderate threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(26, 20)  # Warmup for Ichimoku and volume
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or
            np.isnan(senkou_b[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kijun_val = kijun[i]
        tenkan_val = tenkan[i]
        senkou_b_val = senkou_b[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price crosses above Kijun with volume spike and bullish cloud (price > Senkou B)
            if price > kijun_val and tenkan_val > kijun_val and price > senkou_b_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Kijun with volume spike and bearish cloud (price < Senkou B)
            elif price < kijun_val and tenkan_val < kijun_val and price < senkou_b_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below Kijun OR tenkan-kijun cross turns bearish
            if price < kijun_val or tenkan_val < kijun_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above Kijun OR tenkan-kijun cross turns bullish
            if price > kijun_val or tenkan_val > kijun_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Ichimoku_Kijun_Breakout_With_Volume"
timeframe = "12h"
leverage = 1.0