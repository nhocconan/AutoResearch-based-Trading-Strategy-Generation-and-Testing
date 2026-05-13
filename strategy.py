#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d timeframe provide high-probability reversal (R3/S3) and continuation (R4/S4) signals.
- Breakout above R3 with 1d uptrend and volume spike = long
- Breakdown below S3 with 1d downtrend and volume spike = short
- Breakout above R4 or below S4 signals continuation with same filters
- Uses 12h trend filter for additional confirmation
- Target: 12-37 trades/year (50-150 total over 4 years)
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Camarilla pivot levels from 1d (using previous day's HLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R4 = pc + ((ph - pl) * 1.1 / 2)
    R3 = pc + ((ph - pl) * 1.1 / 4)
    R2 = pc + ((ph - pl) * 1.1 / 6)
    R1 = pc + ((ph - pl) * 1.1 / 12)
    S1 = pc - ((ph - pl) * 1.1 / 12)
    S2 = pc - ((ph - pl) * 1.1 / 6)
    S3 = pc - ((ph - pl) * 1.1 / 4)
    S4 = pc - ((ph - pl) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_12h = df_12h['close'].values > ema_20_12h
    downtrend_12h = df_12h['close'].values < ema_20_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        uptrend = uptrend_12h_aligned[i]
        downtrend = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3 (reversal) or R4 (continuation) with 12h uptrend and volume
            if ((close[i] > r3 or close[i] > r4) and uptrend and vol_conf):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 (reversal) or S4 (continuation) with 12h downtrend and volume
            elif ((close[i] < s3 or close[i] < s4) and downtrend and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to R3 or trend turns down
            if close[i] < r3 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to S3 or trend turns up
            if close[i] > s3 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals