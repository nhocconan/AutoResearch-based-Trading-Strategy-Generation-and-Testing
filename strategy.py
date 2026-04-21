#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use 1d Camarilla pivot levels (S1/R1) for breakout entries with 1d EMA34 trend filter and volume confirmation.
Designed to capture breakouts in the direction of the daily trend. Works in bull/bear markets by following higher timeframe trend while using pivot levels for entry timing.
Target 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    # Camarilla levels: S1 = close - (range * 1.0833), R1 = close + (range * 1.0833)
    s1_1d = close_1d - (range_1d * 1.08333)
    r1_1d = close_1d + (range_1d * 1.08333)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # === 1d trend filter: 34-period EMA ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(s1_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        s1 = s1_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        trend = ema_34_1d_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Close breaks above R1 + volume spike > 1.5 + price above 1d EMA34
            if (price_close > r1 and 
                vol_spike > 1.5 and 
                price_close > trend):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + volume spike > 1.5 + price below 1d EMA34
            elif (price_close < s1 and 
                  vol_spike > 1.5 and 
                  price_close < trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price crosses 1d EMA34 in opposite direction
            if position == 1 and price_close < trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0