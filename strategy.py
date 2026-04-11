#!/usr/bin/env python3
# 12h_1w_kama_rsi_vol_v1
# Strategy: 12h KAMA trend direction with RSI momentum and volume confirmation, 1w trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals. In trending markets,
# KAMA follows price closely; in ranging markets, it stays flat. Combined with RSI for momentum
# confirmation and weekly trend filter to avoid counter-trend trades, this should capture major
# moves while minimizing whipsaws. Low frequency (~15-30/year) to reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_kama_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_avg_30 = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (1.5 * vol_avg_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: KAMA direction + RSI momentum + volume + trend alignment
        if (close[i] > kama[i] and  # Price above KAMA (uptrend)
            rsi[i] > 50 and rsi[i] > rsi[i-1] and  # RSI > 50 and rising
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < kama[i] and  # Price below KAMA (downtrend)
              rsi[i] < 50 and rsi[i] < rsi[i-1] and  # RSI < 50 and falling
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend change or momentum loss
        elif position == 1 and (close[i] <= kama[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= kama[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals