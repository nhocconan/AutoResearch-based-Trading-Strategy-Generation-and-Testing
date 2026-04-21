#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Trend
Hypothesis: Use weekly trend (EMA34) to filter daily KAMA trend signals. Long when KAMA > price and price > weekly EMA34, short when KAMA < price and price < weekly EMA34. RSI(14) confirms momentum (>50 for long, <50 for short). Designed for 1d timeframe to limit trade frequency (target: 10-30/year) and reduce fee drift. Works in bull markets by buying dips in uptrend and in bear markets by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False).values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Efficiency Ratio (ER)
    change = np.abs(np.abs(close - np.roll(close, 10)))
    volatility = np.sum(np.abs(np.diff(np.concatenate([[-np.nan], close[:-1]]))), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    price_change = np.abs(close - np.roll(close, 10))
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # wait for weekly EMA34 to be ready
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: KAMA > price (uptrend), price > weekly EMA34 (weekly uptrend), RSI > 50 (bullish momentum)
            if kama[i] > price and price > ema_34_1w_aligned[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA < price (downtrend), price < weekly EMA34 (weekly downtrend), RSI < 50 (bearish momentum)
            elif kama[i] < price and price < ema_34_1w_aligned[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA < price (trend change) or weekly EMA34 cross below price (weekly trend change)
            if kama[i] < price or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA > price (trend change) or weekly EMA34 cross above price (weekly trend change)
            if kama[i] > price or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_RSI_Trend"
timeframe = "1d"
leverage = 1.0