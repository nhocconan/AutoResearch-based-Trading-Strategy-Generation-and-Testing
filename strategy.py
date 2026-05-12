#!/usr/bin/env python3
# 1D_KAMA_REVERSAL_FILTER
# Hypothesis: KAMA adapts to market efficiency, capturing trend direction while filtering noise.
# Long when price crosses above KAMA with rising ADX (trend confirmation); short when price crosses below KAMA with rising ADX.
# Exit when price returns to KAMA or ADX falls below threshold (range market).
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Designed for low-frequency, high-conviction trades on daily timeframe to minimize fee drag.
# Targets 15-25 trades/year on BTC/ETH, focusing on major trend shifts.

name = "1D_KAMA_REVERSAL_FILTER"
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
    
    # KAMA (Adaptive Moving Average) - 20 periods
    # Efficiency Ratio: |close - close[9]| / sum(|close - close[-1]| over 10 periods)
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.subtract(close[1:], close[:-1]))  # sum |close[t] - close[t-1]|
                       .reshape(-1, 1), axis=1)  # reshape for broadcasting
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align lengths
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    sc = (er * 0.603 + 0.0645) ** 2  # scaled to [0.0645, 0.667]
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ADX (14-period) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high - low
    tr2 = np.abs(np.subtract(high[1:], close[:-1]))
    tr3 = np.abs(np.subtract(low[1:], close[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros(n)
    for i in range(1, n):
        tr_val = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr_val if i < 14 else (atr[i-1] * 13 + tr_val) / 14  # Wilder's smoothing
    plus_di = 100 * (np.convolve(plus_dm, np.ones(14)/14, mode='same') / (atr + 1e-10))
    minus_di = 100 * (np.convolve(minus_dm, np.ones(14)/14, mode='same') / (atr + 1e-10))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.zeros(n)
    for i in range(14, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14 if not np.isnan(dx[i]) else adx[i-1]
    
    # Weekly trend filter: 1-week EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    pclose_1w = df_1w['close'].values
    ema_1w = pd.Series(pclose_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price crosses above KAMA with rising ADX and above weekly EMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and adx[i] > adx[i-1] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA with rising ADX and below weekly EMA
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and adx[i] > adx[i-1] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to KAMA or ADX weakens (range market)
            if close[i] < kama[i] or adx[i] < 20:  # ADX < 20 indicates ranging
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to KAMA or ADX weakens
            if close[i] > kama[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals