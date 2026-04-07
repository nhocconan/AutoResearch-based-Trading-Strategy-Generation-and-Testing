#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Spike
Hypothesis: Ichimoku TK cross with cloud filter identifies trend strength.
Using 1d EMA200 as trend filter ensures alignment with higher timeframe trend.
Volume spikes confirm institutional participation. Works in bull/bear by following trend.
Target: 15-30 trades/year per symbol to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku Components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Volume Spike Detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA200 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR TK cross turns bearish
            if (close[i] < cloud_bottom or 
                (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR TK cross turns bullish
            if (close[i] > cloud_top or 
                (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish TK cross above cloud + price above 1d EMA200 + volume spike
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and
                close[i] > cloud_top and 
                close[i] > ema_200_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Bearish TK cross below cloud + price below 1d EMA200 + volume spike
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and
                  close[i] < cloud_bottom and 
                  close[i] < ema_200_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals