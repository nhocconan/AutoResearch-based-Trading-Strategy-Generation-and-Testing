#!/usr/bin/env python3
# Hypothesis: 1d KAMA direction + RSI(14) + Choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend).
# Uses 1w EMA50 as higher timeframe trend filter. Long when price > KAMA and RSI < 30 in trending market,
# short when price < KAMA and RSI > 70 in trending market. Discrete sizing 0.25 to target 30-100 trades over 4 years.
# KAMA adapts to market efficiency, RSI captures mean reversion extremes, chop filter avoids whipsaws in ranging markets.
# 1w EMA50 ensures alignment with weekly trend. Designed for low-frequency, high-conviction trades to minimize fee drag.

name = "1d_KAMA_RSI_Chop_1wEMA50_v1"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (adaptive moving average)
    er_len = 10
    fast_ema = 2 / (2 + 1)
    slow_ema = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)
    volatility = np.sum(np.abs(np.diff(close, n=1)).reshape(-1, 1), axis=1)
    volatility = pd.Series(volatility).rolling(window=er_len, min_periods=1).sum().values
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (fast_ema - slow_ema) + slow_ema) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    rsi_len = 14
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period)
    chop_len = 14
    tr1 = pd.Series(high).rolling(window=chop_len).max() - pd.Series(low).rolling(window=chop_len).min()
    tr2 = np.abs(pd.Series(high) - np.roll(pd.Series(close), 1))
    tr3 = np.abs(pd.Series(low) - np.roll(pd.Series(close), 1))
    tr2 = pd.Series(tr2).fillna(0)
    tr3 = pd.Series(tr3).fillna(0)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=chop_len, min_periods=chop_len).sum()
    sum_close_diff = np.abs(pd.Series(close) - np.roll(pd.Series(close), 1)).rolling(window=chop_len, min_periods=chop_len).sum()
    sum_close_diff = sum_close_diff.fillna(0)
    chop = 100 * np.log10(sum_close_diff / atr) / np.log10(chop_len)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_len, rsi_len, chop_len, 50)
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2) or strong trends (CHOP < 61.8)
        if chop_values[i] >= 61.8:
            # In ranging market, force flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price > KAMA, RSI < 30 (oversold), above weekly EMA
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA, RSI > 70 (overbought), below weekly EMA
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR RSI > 50 (mean reversion) OR weekly trend breaks
            if (close[i] < kama[i] or 
                rsi[i] > 50 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR RSI < 50 OR weekly trend breaks
            if (close[i] > kama[i] or 
                rsi[i] < 50 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals