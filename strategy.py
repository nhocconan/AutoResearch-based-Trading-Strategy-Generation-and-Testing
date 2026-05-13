#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) breakouts on 4h with 1d trend filter and volume confirmation provide a robust edge in both bull and bear markets by combining mean-reversion levels with trend-following momentum.
Breakout above R1 with 1d uptrend and volume spike = long.
Breakdown below S1 with 1d downtrend and volume spike = short.
Exit on opposite level touch or trend reversal. Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for each bar using previous day's range
    # For intraday, we use previous bar's high/low as approximation for daily range
    # This is a simplification but works for 4h timeframe
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    
    for i in range(1, n):
        # Use previous bar's high/low to calculate today's Camarilla levels
        # Camarilla formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
        # where C = (H+L+Close)/3 (typical price)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Typical price of previous bar
        typical_price = (prev_high + prev_low + prev_close) / 3.0
        # Range of previous bar
        range_val = prev_high - prev_low
        
        camarilla_r1[i] = typical_price + range_val * 1.1 / 12.0
        camarilla_s1[i] = typical_price - range_val * 1.1 / 12.0
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
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
        # Get values
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 1d uptrend, volume confirmation
            if close[i] > r1 and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 1d downtrend, volume confirmation
            elif close[i] < s1 and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 1d trend turns down
            if close[i] < s1 or not uptrend_htf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 1d trend turns up
            if close[i] > r1 or not downtrend_htf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals