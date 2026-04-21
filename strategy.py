#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_Breakout_VolumeFilter_V2
Hypothesis: Breakout of Camarilla R1/S1 levels on 12h timeframe with 1d trend filter (EMA34) and volume confirmation.
Works in bull/bear: In uptrend, buy R1 breakout; in downtrend, sell S1 breakout. Uses 1d EMA34 for trend, volume spike for confirmation.
Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1 (primary breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: price > R1 (breakout) AND 1d uptrend AND volume
            if (price > r1_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and  # 1d EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S1 (breakdown) AND 1d downtrend AND volume
            elif (price < s1_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and  # 1d EMA falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 1d EMA34 (trend reversal) or price < S1 (mean reversion)
            if (price < ema_34_1d_aligned[i] or 
                price < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 1d EMA34 (trend reversal) or price > R1 (mean reversion)
            if (price > ema_34_1d_aligned[i] or 
                price > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_Breakout_VolumeFilter_V2"
timeframe = "12h"
leverage = 1.0