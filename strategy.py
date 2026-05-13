#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_WeeklyTrend_Volume
Hypothesis: Camarilla R3/S3 breakouts with weekly trend filter and volume confirmation work in both bull and bear markets.
Breakout above R3 with weekly uptrend and volume spike = long.
Breakdown below S3 with weekly downtrend and volume spike = short.
Exit on opposite Camarilla level (R2/S2) touch or trend reversal. Uses weekly trend for higher timeframe bias.
Target: 10-25 trades/year per symbol.
"""

name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Camarilla levels (based on previous day's range)
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.25 * (high - low)
    # R2 = close + 1.166 * (high - low)
    # R1 = close + 1.083 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 1.083 * (high - low)
    # S2 = close - 1.166 * (high - low)
    # S3 = close - 1.25 * (high - low)
    # S4 = close - 1.5 * (high - low)
    
    # Calculate previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # first bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels
    R3 = prev_close + 1.25 * (prev_high - prev_low)
    S3 = prev_close - 1.25 * (prev_high - prev_low)
    R2 = prev_close + 1.166 * (prev_high - prev_low)
    S2 = prev_close - 1.166 * (prev_high - prev_low)
    
    # Weekly trend filter (HTF: 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r3 = R3[i]
        s3 = S3[i]
        r2 = R2[i]
        s2 = S2[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3, weekly uptrend, volume confirmation
            if close[i] > r3 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, weekly downtrend, volume confirmation
            elif close[i] < s3 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch R2 or weekly trend turns down
            if close[i] < r2 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch S2 or weekly trend turns up
            if close[i] > s2 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals