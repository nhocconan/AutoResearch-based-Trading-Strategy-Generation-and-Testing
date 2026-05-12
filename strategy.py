#!/usr/bin/env python3
# 160090: 1d_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction. Combined with RSI extremes and Choppiness Index regime filter, this strategy captures strong trends while avoiding choppy markets. Works in both bull and bear by following adaptive trend with strict entry filters. Uses daily timeframe with weekly trend context for lower trade frequency and reduced fee drag.

name = "1d_KAMA_Trend_RSI_Chop_Filter"
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

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)

    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Pad beginning with zeros
    er = np.concatenate([np.full(er_length, np.nan), er])

    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.concatenate([np.full(er_length, np.nan), sc])

    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]  # Seed
    for i in range(er_length + 1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)

    # Weekly trend filter: EMA34 on weekly close
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])

    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values

    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low

    chop = np.where(range_hl != 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    chop = np.concatenate([np.full(13, np.nan), chop])

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after warmup period
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI < 30 (oversold) + Chop > 61.8 (ranging) + weekly uptrend + volume
            if (close[i] > kama_aligned[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI > 70 (overbought) + Chop > 61.8 (ranging) + weekly downtrend + volume
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals