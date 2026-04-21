#!/usr/bin/env python3
"""
12h_1d_Keltner_Channel_Breakout_Volume
Hypothesis: Use 1d Keltner Channel breakouts with volume confirmation on 12h timeframe.
Long when price breaks above upper KC with volume > 1.5x 20-bar avg.
Short when price breaks below lower KC with volume > 1.5x 20-bar avg.
Exit when price returns to middle line (EMA).
Designed for 12h timeframe to capture multi-day trends with ~15-35 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume filter reduces false breakouts and whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Keltner Channel parameters
    kc_period = 20
    kc_multiplier = 2.0
    
    # Calculate EMA (middle line)
    ema = pd.Series(close_1d).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Calculate ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Upper and lower channels
    upper_kc = ema + (kc_multiplier * atr)
    lower_kc = ema - (kc_multiplier * atr)
    
    # Align to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema)
    upper_kc_aligned = align_htf_to_ltf(prices, df_1d, upper_kc)
    lower_kc_aligned = align_htf_to_ltf(prices, df_1d, lower_kc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_aligned[i]) or np.isnan(upper_kc_aligned[i]) or 
            np.isnan(lower_kc_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above upper KC + volume confirmation
            if price > upper_kc_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower KC + volume confirmation
            elif price < lower_kc_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle line (EMA)
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle line (EMA)
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Keltner_Channel_Breakout_Volume"
timeframe = "12h"
leverage = 1.0