#!/usr/bin/env python3
"""
4h_Momentum_With_Volume_Trend_Filter
Hypothesis: RSI(14) momentum with volume confirmation and 1d trend filter provides robust entries in both bull and bear markets.
Long when RSI crosses above 50 with rising volume and 1d uptrend; short when RSI crosses below 50 with rising volume and 1d downtrend.
Exit on opposite RSI cross or trend failure. Uses 4h EMA for trend confirmation.
Target: 25-35 trades/year per symbol.
"""

name = "4h_Momentum_With_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    # 4h trend: EMA20
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_4h = close > ema_20
    downtrend_4h = close < ema_20
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1d = df_1d['close'].values > ema_20_1d
    downtrend_1d = df_1d['close'].values < ema_20_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        vol_conf = volume_conf[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        
        if position == 0:
            # LONG: RSI crosses above 50 with volume confirmation and uptrend on both timeframes
            if rsi_prev <= 50 and rsi_now > 50 and vol_conf and uptrend and uptrend_htf:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 50 with volume confirmation and downtrend on both timeframes
            elif rsi_prev >= 50 and rsi_now < 50 and vol_conf and downtrend and downtrend_htf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses below 50 or 4h trend turns down
            if rsi_prev >= 50 and rsi_now < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses above 50 or 4h trend turns up
            if rsi_prev <= 50 and rsi_now > 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals