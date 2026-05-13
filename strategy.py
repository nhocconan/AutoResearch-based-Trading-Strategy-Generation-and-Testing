#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_WeeklyTrend_DailyVolume
Hypothesis: Camarilla pivot breakouts on 12h timeframe with weekly trend filter and daily volume confirmation work in both bull and bear markets.
Breakout above R1 with weekly uptrend and volume spike = long.
Breakdown below S1 with weekly downtrend and volume spike = short.
Exit on opposite S1/R1 touch or weekly trend reversal. Uses daily volume spike for confirmation.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

name = "12h_Camarilla_R1_S1_Breakout_WeeklyTrend_DailyVolume"
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
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = prev_close + (range_hl * 1.1 / 12.0)
    s1 = prev_close - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly trend filter: EMA50 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Daily volume confirmation: volume > 2.0 * 20-day average
    vol_20 = np.zeros(n)
    for i in range(20, n):
        vol_20[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Get values for current bar
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, weekly uptrend, volume confirmation
            if close[i] > r1_val and uptrend_weekly and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, weekly downtrend, volume confirmation
            elif close[i] < s1_val and downtrend_weekly and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or weekly trend turns down
            if close[i] < s1_val or not uptrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or weekly trend turns up
            if close[i] > r1_val or not downtrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals