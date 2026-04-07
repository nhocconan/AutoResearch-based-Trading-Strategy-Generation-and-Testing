#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
Bull Power = High - EMA13, Bear Power = EMA13 - Low.
In bull market (1d close > EMA50): long when Bull Power crosses above zero with volume confirmation.
In bear market (1d close < EMA50): short when Bear Power crosses above zero with volume confirmation.
Uses Elder Ray to detect institutional buying/selling pressure, filtered by daily trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_trend_volume_v1"
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === ELDER RAY (LTF) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # warmup for EMA13 and volume EMA
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power crosses above zero OR trend turns bearish
            if bear_power[i] > 0 or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power crosses above zero OR trend turns bullish
            if bull_power[i] > 0 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend
            if bull_trend:
                # In bull market: long when Bull Power crosses above zero
                if bull_power[i] > 0 and bull_power[i-1] <= 0:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short when Bear Power crosses above zero
                if bear_power[i] > 0 and bear_power[i-1] <= 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals