#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from daily data with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R3 with uptrend and volume spike = long.
Breakdown below S3 with downtrend and volume spike = short.
Exit on opposite level touch or trend reversal. Target: 20-50 trades/year per symbol.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R3 = close + (high - low) * 1.1/2, S3 = close - (high - low) * 1.1/2
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.1 / 2
    s3 = close_1d - camarilla_range * 1.1 / 2
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3, 4h uptrend, 1d uptrend filter, volume confirmation
            if close[i] > r3_val and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3, 4h downtrend, 1d downtrend filter, volume confirmation
            elif close[i] < s3_val and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S3 or 4h trend turns down
            if close[i] < s3_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: touch R3 or 4h trend turns up
            if close[i] > r3_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals