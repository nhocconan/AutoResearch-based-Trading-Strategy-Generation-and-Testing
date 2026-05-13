#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from 1d, with 1d trend filter and volume confirmation, work on 12h timeframe.
Breakout above R1 with uptrend and volume spike = long.
Breakdown below S1 with downtrend and volume spike = short.
Exit on opposite level touch or trend reversal.
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
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous day's range
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid division by zero and handle NaN
    range_hl = prev_high - prev_low
    r1 = prev_close + 1.1 * range_hl / 12
    s1 = prev_close - 1.1 * range_hl / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            signals[i] = 0.0
            continue
            
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 1d uptrend, volume confirmation
            if close[i] > r1_val and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 1d downtrend, volume confirmation
            elif close[i] < s1_val and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 1d trend turns down
            if close[i] < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 1d trend turns up
            if close[i] > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals