#/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R1 with 1d uptrend and volume spike = long.
Breakdown below S1 with 1d downtrend and volume spike = short.
Exit on opposite level touch or trend reversal.
Uses 12h timeframe with 1d trend filter to reduce trade frequency and avoid overtrading.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
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
    
    # Camarilla levels from previous day (using previous close)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    cam_high = np.concatenate([[high[0]], high[:-1]])  # previous high
    cam_low = np.concatenate([[low[0]], low[:-1]])    # previous low
    
    # Calculate Camarilla levels
    range_prev = cam_high - cam_low
    r1 = prev_close + 1.1 * range_prev * 1.0 / 12
    s1 = prev_close - 1.1 * range_prev * 1.0 / 12
    r3 = prev_close + 1.1 * range_prev * 3.0 / 12
    s3 = prev_close - 1.1 * range_prev * 3.0 / 12
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        r1_val = r1[i]
        s1_val = s1[i]
        r3_val = r3[i]
        s3_val = s3[i]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1 with 1d uptrend and volume confirmation
            if close[i] > r1_val and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with 1d downtrend and volume confirmation
            elif close[i] < s1_val and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S3 or 1d trend turns down
            if close[i] < s3_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R3 or 1d trend turns up
            if close[i] > r3_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals