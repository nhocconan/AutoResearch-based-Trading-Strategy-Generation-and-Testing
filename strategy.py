#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R1 with uptrend and volume spike = long.
Breakdown below S1 with downtrend and volume spike = short.
Exit on close back inside R1-S1 range or trend reversal. Uses 1d trend filter for higher timeframe bias.
Target: 12-37 trades/year per symbol.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous day's range
    # For each bar, we use the previous day's high/low/close
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # True range for the previous day
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R1 = prev_close + (prev_range * 1.1 / 12)
    S1 = prev_close - (prev_range * 1.1 / 12)
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if Camarilla levels are not available (first bar)
        if np.isnan(R1[i]) or np.isnan(S1[i]):
            signals[i] = 0.0
            continue
            
        r1 = R1[i]
        s1 = S1[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 1d uptrend, volume confirmation
            if close[i] > r1 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 1d downtrend, volume confirmation
            elif close[i] < s1 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close back inside R1-S1 range or 1d trend turns down
            if close[i] < r1 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close back inside R1-S1 range or 1d trend turns up
            if close[i] > s1 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals