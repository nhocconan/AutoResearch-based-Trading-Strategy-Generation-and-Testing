#!/usr/bin/env python3
"""
6h_1d_Camarilla_R1S1_Breakout_Volume_Control_v2
Hypothesis: Use 1d Camarilla R1/S1 levels on 6h timeframe with volume confirmation and EMA34 trend filter.
Long when price breaks above 1d R1 with volume > 1.8x 20-period average and EMA34 > EMA89.
Short when price breaks below 1d S1 with volume > 1.8x 20-period average and EMA34 < EMA89.
Exit when price crosses 1d pivot point. Higher volume threshold reduces false breaks.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear by following higher timeframe trend via EMA34/EMA89 crossover.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
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
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate EMA34 and EMA89 on price data
    close = prices['close'].values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89 = pd.Series(close).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema34[i]) or np.isnan(ema89[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.8 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: EMA34 > EMA89 for uptrend, EMA34 < EMA89 for downtrend
        uptrend = ema34[i] > ema89[i]
        downtrend = ema34[i] < ema89[i]
        
        if position == 0:
            # Long conditions: break above R1 + volume + uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + downtrend
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Control_v2"
timeframe = "6h"
leverage = 1.0