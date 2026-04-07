#!/usr/bin/env python3
"""
6h Elder Ray with 12h Trend Filter and Volume Spike
Hypothesis: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure.
Using 12h EMA50 as trend filter provides stronger trend identification.
Volume spikes confirm institutional participation.
This should work in both bull and bear regimes by following the trend.
Target: 20-50 trades/year per symbol to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_12h_trend_volume_v1"
timeframe = "6h"
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
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume Spike Detector
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # 12h EMA50 Trend Filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        if np.isnan(ema_13[i]) or np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative (selling pressure)
            if bear_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive (buying pressure)
            if bull_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: Bull Power positive (buying pressure) + price above 12h EMA50 + volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power negative (selling pressure) + price below 12h EMA50 + volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals