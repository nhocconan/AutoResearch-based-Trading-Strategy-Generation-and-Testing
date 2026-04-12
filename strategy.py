#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_filter_v1
Hypothesis: Daily strategy using KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI identifies overbought/oversold conditions.
Choppiness Index filters out choppy regimes (CHOP > 61.8) where trend following fails, focusing on trending markets.
Works in bull/bear by only taking trades aligned with the weekly trend (KAMA direction) and avoiding false signals in chop.
Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend and Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly KAMA for trend direction (adaptive smoothing)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first 'length' elements
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.maximum(volatility[length-1:], 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(close_1w, length=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Weekly Choppiness Index (14-period)
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Pad to same length as close
        tr = np.concatenate([[np.nan], tr])
        # Sum of TR over period
        atr_sum = np.nansum(tr.reshape(-1, length), axis=1) if len(tr) >= length else np.full(len(close), np.nan)
        atr_sum = np.concatenate([np.full(length-1, np.nan), atr_sum])
        # Highest high and lowest low over period
        hh = np.maximum.accumulate(high)
        ll = np.minimum.accumulate(low)
        # For rolling window, use pandas for simplicity
        hh_series = pd.Series(high)
        ll_series = pd.Series(low)
        tr_series = pd.Series(tr)
        roll_max = hh_series.rolling(window=length, min_periods=length).max().values
        roll_min = ll_series.rolling(window=length, min_periods=length).min().values
        roll_sum = tr_series.rolling(window=length, min_periods=length).sum().values
        # Chop = 100 * log10(sum(tr)/(max(high)-min(low))) / log10(length)
        with np.errstate(divide='ignore', invalid='ignore'):
            chop = 100 * np.log10(roll_sum / np.maximum(roll_max - roll_min, 1e-10)) / np.log10(length)
        return chop
    
    chop_1w = calculate_chop(high_1w, low_1w, close_1w, length=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Daily RSI (14-period)
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # First average gain/loss
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        # Wilder smoothing
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > KAMA (uptrend) AND RSI < 30 (oversold) AND CHOP < 61.8 (trending)
        if (close[i] > kama_1w_aligned[i] and rsi[i] < 30 and chop_1w_aligned[i] < 61.8 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < KAMA (downtrend) AND RSI > 70 (overbought) AND CHOP < 61.8 (trending)
        elif (close[i] < kama_1w_aligned[i] and rsi[i] > 70 and chop_1w_aligned[i] < 61.8 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or RSI crosses back to neutral (40-60) or CHOP > 61.8 (choppy)
        elif position == 1 and (rsi[i] > 50 or chop_1w_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 50 or chop_1w_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0