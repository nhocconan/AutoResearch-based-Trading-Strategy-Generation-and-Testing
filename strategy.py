#!/usr/bin/env python3
name = "1d_KAMA_Direction_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA direction on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Volume confirmation: volume > 1.2 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for weekly EMA and KAMA
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up (close > kama) + weekly uptrend + RSI > 50 + volume confirmation
            if close[i] > kama[i] and close[i] > ema34_1w_aligned[i] and rsi[i] > 50 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (close < kama) + weekly downtrend + RSI < 50 + volume confirmation
            elif close[i] < kama[i] and close[i] < ema34_1w_aligned[i] and rsi[i] < 50 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or weekly trend reversal
            if close[i] < kama[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or weekly trend reversal
            if close[i] > kama[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals