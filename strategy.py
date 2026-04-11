#!/usr/bin/env python3
# 12h_1d_kama_rsi_chop_v1
# Strategy: 12h KAMA direction + RSI(14) + Choppiness Index filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market efficiency, trending when ER high, mean-reverting when ER low.
# Combined with RSI for momentum and Choppiness Index to avoid ranging markets (CHOP > 61.8).
# Works in bull/bear: KAMA catches trends, RSI avoids extremes, CHOP filter prevents whipsaws in sideways markets.
# Target: 15-30 trades/year (~60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h KAMA (adaptive moving average)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # Will fix below
    
    # Proper volatility calculation (sum of absolute changes over 10 periods)
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 12h RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # ATR(14)
    atr = np.zeros_like(close)
    atr[13] = np.mean(tr[1:15])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr_14 = np.zeros_like(close)
    for i in range(13, len(close)):
        if i == 13:
            sum_atr_14[i] = np.sum(atr[0:14])
        else:
            sum_atr_14[i] = sum_atr_14[i-1] - atr[i-14] + atr[i]
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(len(close)):
        if i < 14:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
    # Chop = 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    hh_ll = highest_high - lowest_low
    chop = np.where(hh_ll > 0, 100 * np.log10(sum_atr_14 / hh_ll) / np.log10(14), 50)
    
    # 1d RSI for trend filter (avoid counter-trend)
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d)
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = np.zeros_like(close_1d)
    avg_loss_1d = np.zeros_like(close_1d)
    if len(close_1d) > 14:
        avg_gain_1d[14] = np.mean(gain_1d[1:15])
        avg_loss_1d[14] = np.mean(loss_1d[1:15])
        for i in range(15, len(close_1d)):
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
        rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 100)
        rsi_1d = 100 - (100 / (1 + rs_1d))
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    else:
        rsi_1d_aligned = np.full(n, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup for KAMA/RSI/Chop
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Choppiness filter: avoid ranging markets (CHOP > 61.8 = range)
        chop_filter = chop[i] <= 61.8  # Only trade when NOT in strong ranging market
        
        # RSI filter: avoid extremes (only trade when RSI 30-70)
        rsi_filter = (rsi[i] >= 30) & (rsi[i] <= 70)
        
        # Trend direction: price vs KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # 1d RSI trend filter: avoid counter-trend trades
        uptrend_1d = rsi_1d_aligned[i] > 50
        downtrend_1d = rsi_1d_aligned[i] < 50
        
        # Entry conditions
        # Long: Price above KAMA AND RSI in neutral range AND 1d uptrend AND not choppy
        if above_kama and rsi_filter and uptrend_1d and chop_filter and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price below KAMA AND RSI in neutral range AND 1d downtrend AND not choppy
        elif below_kama and rsi_filter and downtrend_1d and chop_filter and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Reverse signal or RSI reaches extreme
        elif position == 1 and (below_kama or rsi[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (above_kama or rsi[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals