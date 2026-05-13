#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_Trend_Volume_1D
Hypothesis: Camarilla R3/S3 breakouts on 4h with 1d trend filter and volume confirmation work in both bull and bear markets.
Breakout above R3 with 1d uptrend and volume spike = long.
Breakdown below S3 with 1d downtrend and volume spike = short.
Exit on opposite Camarilla level touch or trend reversal.
Target: 20-50 trades/year per symbol.
"""

name = "4H_Camarilla_R3S3_Breakout_Trend_Volume_1D"
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
    
    # Calculate daily Camarilla levels from previous day
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    # Camarilla formula: range = prev_high - prev_low
    # R3 = prev_close + 1.1 * range / 2
    # S3 = prev_close - 1.1 * range / 2
    daily_range = prev_high - prev_low
    r3 = prev_close + 1.1 * daily_range / 2
    s3 = prev_close - 1.1 * daily_range / 2
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
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
    
    for i in range(50, n):
        # Get values
        r3_val = r3[i]
        s3_val = s3[i]
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