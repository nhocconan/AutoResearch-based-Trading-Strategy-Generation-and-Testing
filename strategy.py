#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily data act as strong support/resistance. 
Breakout above R1 or below S1 with volume confirmation and 12h EMA50 trend filter captures 
institutional breakouts. Works in bull/bear markets by following 12h trend while using 
Camarilla levels for precise entry/exit. Target 20-40 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h trend filter: 50-period EMA ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Calculate Camarilla levels from previous day ===
    high_prev = df_1d['high'].shift(1).values  # Previous day high
    low_prev = df_1d['low'].shift(1).values    # Previous day low
    close_prev = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla formula
    range_prev = high_prev - low_prev
    # R1 = close + (range * 1.1 / 12)
    # S1 = close - (range * 1.1 / 12)
    r1 = close_prev + (range_prev * 1.1 / 12)
    s1 = close_prev - (range_prev * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        trend_12h = ema_50_12h_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Break above R1 + volume spike > 1.5 + price above 12h EMA50
            if (price_close > r1_level and 
                vol_spike > 1.5 and 
                price_close > trend_12h):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume spike > 1.5 + price below 12h EMA50
            elif (price_close < s1_level and 
                  vol_spike > 1.5 and 
                  price_close < trend_12h):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to opposite Camarilla level
            if position == 1 and price_close < s1_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0