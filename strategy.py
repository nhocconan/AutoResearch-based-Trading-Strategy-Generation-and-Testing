#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_1wTrend_Volume
Hypothesis: Trade breakouts at weekly Camarilla R1/S1 levels on daily timeframe with weekly trend filter and volume confirmation. Uses weekly trend to capture major moves in bull/bear, with daily precision for entries/exits. Targets 20-50 trades over 4 years to minimize fee drag.
"""

name = "1d_Camarilla_R1_S1_1wTrend_Volume"
timeframe = "1d"
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
    
    # === Weekly OHLC for Camarilla Pivots ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's OHLC
    pw = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    
    # Camarilla R1/S1 (most significant levels for breakout)
    camarilla_r1 = pc + (pw - pl) * 1.1 / 2
    camarilla_s1 = pc - (pw - pl) * 1.1 / 2
    
    # Align to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_daily = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # === Weekly Trend Filter (EMA34) ===
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Filter (1.5x 20-period EMA on daily) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or np.isnan(ema34_daily[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with weekly uptrend and volume
            if (close[i] > r1_daily[i] and 
                close[i] > ema34_daily[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with weekly downtrend and volume
            elif (close[i] < s1_daily[i] and 
                  close[i] < ema34_daily[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal)
            if close[i] < s1_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R1 (reversal)
            if close[i] > r1_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals