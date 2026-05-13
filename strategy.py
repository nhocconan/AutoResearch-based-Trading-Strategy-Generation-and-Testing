#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE for Elder Ray and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Elder Ray Power calculation on daily data
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d   # Bull Power = High - EMA13
    bear_power = low_1d - ema13_1d    # Bear Power = Low - EMA13
    
    # Align Elder Ray components to 6H timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6H EMA34 for trend filter (longer-term trend)
    close_s = pd.Series(close)
    ema34_6h = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current volume > 20-period average
    volume_s = pd.Series(volume)
    vol_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to EMA34
        price_above_ema = close[i] > ema34_6h[i]
        price_below_ema = close[i] < ema34_6h[i]
        
        if position == 0:
            # LONG: Strong bull power + price above EMA34 + volume confirmation
            if (bull_power_aligned[i] > 0) and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Strong bear power + price below EMA34 + volume confirmation
            elif (bear_power_aligned[i] < 0) and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative or price closes below EMA34
            if (bull_power_aligned[i] <= 0) or (close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns positive or price closes above EMA34
            if (bear_power_aligned[i] >= 0) or (close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals