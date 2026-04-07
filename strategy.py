#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray with 1d regime filter and volume confirmation.
In bull regime (1d close > 1d EMA100): Elder Ray Bull Power > 0 signals long.
In bear regime (1d close < 1d EMA100): Elder Ray Bear Power < 0 signals short.
Volume must be above 20-period average to confirm strength.
Elder Ray measures bull/bear power as EMA13 of (high - EMA13) and (low - EMA13).
This captures institutional buying/selling pressure with trend alignment.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_regime_volume_v1"
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
    
    # === 1D REGIME FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === ELDER RAY CALCULATION (LTF) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    # Smooth with EMA13 to get the actual Elder Ray indicators
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d EMA
        bull_regime = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Elder Ray Bull Power turns negative OR regime turns bearish
            if bull_power_smooth[i] <= 0 or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray Bear Power turns positive OR regime turns bullish
            if bear_power_smooth[i] >= 0 or bull_regime:
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
                # In bull regime: long when Bull Power > 0
                if bull_power_smooth[i] > 0:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear regime: short when Bear Power < 0
                if bear_power_smooth[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals