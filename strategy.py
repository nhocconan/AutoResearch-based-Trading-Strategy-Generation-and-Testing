#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot R1/S1 breakouts with 12h trend (EMA50) and volume confirmation work in both bull and bear markets.
Breakout above R1 with uptrend and volume spike = long.
Breakdown below S1 with downtrend and volume spike = short.
Exit on opposite S1/R1 touch or trend reversal. Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
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
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 12h trend filter (HTF) - using EMA50 on 12h closes
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r1_val = r1[i]
        s1_val = s1[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_12h_aligned[i]
        downtrend_htf = downtrend_12h_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 4h uptrend, 12h uptrend filter, volume confirmation
            if close[i] > r1_val and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1, 4h downtrend, 12h downtrend filter, volume confirmation
            elif close[i] < s1_val and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 4h trend turns down
            if close[i] < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R1 or 4h trend turns up
            if close[i] > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals