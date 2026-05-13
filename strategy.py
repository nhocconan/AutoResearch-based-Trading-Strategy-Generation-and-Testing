#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: KAMA direction + RSI momentum + volume confirmation works in both bull and bear markets.
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
RSI confirms momentum strength, volume validates breakout strength.
Target: 10-25 trades/year per symbol.
"""

name = "1d_KAMA_Trend_With_RSI_Momentum"
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
    
    # KAMA: Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_dir = kama > np.roll(kama, 1)  # upward slope
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > 1.5 * vol_ma
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema_20_1w
    downtrend_1w = df_1w['close'].values < ema_20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        kama_up = kama_dir[i]
        rsi_val = rsi[i]
        vol_conf = volume_conf[i]
        uptrend_weekly = uptrend_1w_aligned[i]
        downtrend_weekly = downtrend_1w_aligned[i]
        
        if position == 0:
            # LONG: KAMA up, RSI > 50, volume confirmation, weekly uptrend
            if kama_up and rsi_val > 50 and vol_conf and uptrend_weekly:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, volume confirmation, weekly downtrend
            elif not kama_up and rsi_val < 50 and vol_conf and downtrend_weekly:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA down or RSI < 40
            if not kama_up or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA up or RSI > 60
            if kama_up or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals