#!/usr/bin/env python3
"""
12h_Price_Action_Rebound_With_Volume
Hypothesis: Buy at strong support (previous day low) in uptrend, sell at resistance (previous day high) in downtrend.
Works in bull markets by buying dips to support in uptrends and in bear markets by selling rallies to resistance in downtrends.
Volume confirmation filters weak moves. Uses 1-day levels for structure.
Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's high and low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align previous day's levels to 12h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if indicators not ready
        if (np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        high_level = high_1d_aligned[i]
        low_level = low_1d_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price at or above support (prev day low) + uptrend + volume
            if (price_close >= low_level and
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price at or below resistance (prev day high) + downtrend + volume
            elif (price_close <= high_level and
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price reaches opposite level
            if position == 1 and price_close < low_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > high_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Price_Action_Rebound_With_Volume"
timeframe = "12h"
leverage = 1.0