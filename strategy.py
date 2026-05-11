#!/usr/bin/env python3
"""
1d_1w_382_Camarilla_R1S1_Retest_WeeklyTrend
Hypothesis: Trade pullbacks to Camarilla R1/S1 levels after breakouts in the direction of the weekly trend, with volume confirmation. 1d timeframe, 1w trend filter. Target: 15-25 trades/year (60-100 total over 4 years). Works in bull by buying pullbacks in uptrends, in bear by selling rallies in downtrends.
"""

name = "1d_1w_382_Camarilla_R1S1_Retest_WeeklyTrend"
timeframe = "1d"
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
    
    # === Weekly OHLC for Camarilla Pivots (from previous week) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's OHLC
    ph_w = df_1w['high'].values
    pl_w = df_1w['low'].values
    pc_w = df_1w['close'].values
    
    # Camarilla R1 and S1 (most significant levels for retest)
    camarilla_r1_w = pc_w + (ph_w - pl_w) * 1.1 / 2
    camarilla_s1_w = pc_w - (ph_w - pl_w) * 1.1 / 2
    
    # Align to 1d timeframe (wait for weekly bar to close)
    r1_1d = align_htf_to_ltf(prices, df_1w, camarilla_r1_w)
    s1_1d = align_htf_to_ltf(prices, df_1w, camarilla_s1_w)
    
    # === Weekly Trend Filter (EMA34) ===
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Volume Filter (1.5x 20-period EMA on 1d) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly calculations)
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(ema34_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long retest: price pulls back to R1 in uptrend with volume
            if (close[i] >= r1_1d[i] * 0.998 and close[i] <= r1_1d[i] * 1.002 and  # within 0.2% of R1
                close[i] > ema34_1d[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short retest: price rallies to S1 in downtrend with volume
            elif (close[i] >= s1_1d[i] * 0.998 and close[i] <= s1_1d[i] * 1.002 and  # within 0.2% of S1
                  close[i] < ema34_1d[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks above R1 (breakout continuation) or drops below S1 (reversal)
            if close[i] > r1_1d[i] * 1.005:  # break above R1 with buffer
                signals[i] = 0.25  # maintain for momentum
            elif close[i] < s1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks below S1 (breakdown continuation) or rises above R1 (reversal)
            if close[i] < s1_1d[i] * 0.995:  # break below S1 with buffer
                signals[i] = -0.25  # maintain for momentum
            elif close[i] > r1_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals