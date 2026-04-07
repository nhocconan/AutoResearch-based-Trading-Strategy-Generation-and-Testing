#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d regime filter and volume confirmation.
Uses daily EMA13 for trend regime (bull/bear filter) and 6h Elder Ray for entry.
In bull regime (close > daily EMA13): long when Bull Power > 0 and rising.
In bear regime (close < daily EMA13): short when Bear Power < 0 and falling.
Volume must be above 20-period average to confirm.
Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_regime_volume_v1"
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
    
    # === DAILY REGIME FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)  # already shifted
    
    # === 6h ELDER RAY (LTF) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        if np.isnan(daily_ema_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine regime from daily EMA
        bull_regime = close[i] > daily_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR regime turns bearish
            if bull_power[i] <= 0 or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR regime turns bullish
            if bear_power[i] >= 0 or bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on regime
            if bull_regime:
                # In bull regime: long when Bull Power > 0 and rising
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear regime: short when Bear Power < 0 and falling
                if bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals