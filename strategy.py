#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Trade breakouts at Camarilla R1/S1 levels using 4h trend filter and volume confirmation on 1h timeframe. Uses daily pivot levels for structure, 4h EMA for trend, and volume spike for confirmation. Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear by aligning with 4h trend direction.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # === Daily OHLC for Camarilla Pivots (structure) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla R1/S1 from previous day
    camarilla_r1 = pc + (ph - pl) * 1.1 / 2
    camarilla_s1 = pc - (ph - pl) * 1.1 / 2
    
    # Align daily pivots to 1h
    r1_1h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 4h Trend Filter (EMA34) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === Volume Filter (2.0x 24-period EMA on 1h) ===
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_ok = volume > vol_ema24 * 2.0
    
    # === Session Filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily and 4h calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema34_1h[i]) 
            or np.isnan(volume_ok[i]) or np.isnan(session_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_ok[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with 4h uptrend and volume
            if (close[i] > r1_1h[i] and 
                close[i] > ema34_1h[i] and 
                volume_ok[i]):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with 4h downtrend and volume
            elif (close[i] < s1_1h[i] and 
                  close[i] < ema34_1h[i] and 
                  volume_ok[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal)
            if close[i] < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 (reversal)
            if close[i] > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals