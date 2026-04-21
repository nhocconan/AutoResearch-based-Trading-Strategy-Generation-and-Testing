#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1S1_Breakout_Volume_Momentum_v1
Hypothesis: Use 12h timeframe for trend direction (trend and regime) and 4h for entry timing.
Long when price breaks above 12h R1 with 12h uptrend (price > EMA50) and volume > 1.5x 20-period average.
Short when price breaks below 12h S1 with 12h downtrend (price < EMA50) and volume > 1.5x 20-period average.
Exit when price crosses 12h pivot point. Uses volume confirmation to avoid false breaks.
Target: 20-40 trades/year per symbol. Works in bull/bear by following higher timeframe trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    
    # Calculate EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema50_12h_aligned[i])):
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
        
        # 12h trend filter: price > EMA50 for uptrend, price < EMA50 for downtrend
        uptrend = price > ema50_12h_aligned[i]
        downtrend = price < ema50_12h_aligned[i]
        
        if position == 0:
            # Long conditions: break above R1 + volume + 12h uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + 12h downtrend
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

name = "4h_12h_Camarilla_R1S1_Breakout_Volume_Momentum_v1"
timeframe = "4h"
leverage = 1.0