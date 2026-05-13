#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_WeeklyTrend_Volume
Hypothesis: Camarilla pivot breakouts on 12h with weekly trend filter and volume confirmation work in both bull and bear markets.
Breakout above R3 with weekly uptrend and volume spike = long.
Breakdown below S3 with weekly downtrend and volume spike = short.
Exit on opposite touch or weekly trend reversal. Uses 1d volume spike and weekly trend for higher timeframe bias.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

name = "12h_Camarilla_Pivot_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
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
    
    # Camarilla pivot levels from previous day (HLC of previous day)
    # For 12h chart, we use daily OHLC from previous completed day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values  # previous day high
    prev_low = df_1d['low'].shift(1).values    # previous day low
    prev_close = df_1d['close'].shift(1).values  # previous day close
    
    # Calculate Camarilla levels
    # R4 = Close + ((High-Low) * 1.5000)
    # R3 = Close + ((High-Low) * 1.2500)
    # R2 = Close + ((High-Low) * 1.1666)
    # R1 = Close + ((High-Low) * 1.0833)
    # S1 = Close - ((High-Low) * 1.0833)
    # S2 = Close - ((High-Low) * 1.1666)
    # S3 = Close - ((High-Low) * 1.2500)
    # S4 = Close - ((High-Low) * 1.5000)
    
    prev_range = prev_high - prev_low
    r3 = prev_close + (prev_range * 1.2500)
    s3 = prev_close - (prev_range * 1.2500)
    
    # Align Camarilla levels to 12h chart (use previous day's levels for current day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly trend filter: EMA50 on weekly
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 2.0 * 24-period average (2 days of 12h bars)
    vol_ma = np.zeros(n)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3, weekly uptrend, volume confirmation
            if close[i] > r3_val and uptrend_weekly and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, weekly downtrend, volume confirmation
            elif close[i] < s3_val and downtrend_weekly and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S3 or weekly trend turns down
            if close[i] < s3_val or not uptrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R3 or weekly trend turns up
            if close[i] > r3_val or not downtrend_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals