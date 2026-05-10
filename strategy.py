#!/usr/bin/env python3
# 1d_KAMA_Direction_RSI_ChopFilter
# Hypothesis: Daily KAMA trend direction combined with RSI momentum and Choppiness Index regime filter.
# KAMA adapts to market conditions - trending in strong moves, flat in choppy markets.
# RSI provides momentum confirmation while Choppiness Index filters for trending regimes (CHOP < 38.2).
# Designed for low trade frequency (7-25/year) with discrete position sizing to minimize fee drag.
# Works in both bull and bear markets by adapting to trend strength via KAMA and regime filter.

name = "1d_KAMA_Direction_RSI_ChopFilter"
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
    
    # Weekly data for trend filter and regime
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on weekly close
    def calculate_kama(price, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(price, np.nan)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close_1w, period=10, fast=2, slow=30)
    
    # RSI on weekly close
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(price, np.nan)
        avg_loss = np.full_like(price, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close_1w, period=14)
    
    # Choppiness Index on weekly data
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        # True Range sum
        tr_sum = np.zeros(len(close))
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.zeros(len(close))
        ll = np.zeros(len(close))
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.full_like(close, np.nan)
        for i in range(period-1, len(close)):
            if tr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1w, low_1w, close_1w, period=14)
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, trending regime (CHOP < 38.2)
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and chop_aligned[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, trending regime (CHOP < 38.2)
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and chop_aligned[i] < 38.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI < 40 or choppy regime (CHOP > 61.8)
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI > 60 or choppy regime (CHOP > 61.8)
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals