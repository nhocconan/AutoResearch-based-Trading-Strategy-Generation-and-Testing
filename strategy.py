#!/usr/bin/env python3
# 12H_KAMA_RSI_ChopFilter_v2
# Hypothesis: Uses 12h timeframe with KAMA trend filter, RSI mean reversion, and Choppiness Index regime filter.
# Enters long when KAMA indicates uptrend, RSI < 30 (oversold), and market is choppy (CHOP > 61.8).
# Enters short when KAMA indicates downtrend, RSI > 70 (overbought), and market is choppy (CHOP > 61.8).
# Exits when RSI returns to neutral (40-60 range) or trend changes.
# Uses weekly timeframe for higher-order trend confirmation to avoid counter-trend trades.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25 to minimize fee drag.

name = "12H_KAMA_RSI_ChopFilter_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))  # sum of absolute changes
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    atr = np.zeros_like(close)
    tr1 = np.abs(np.subtract(high, low))
    tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first true range
    
    # smoothed TR using Wilder's smoothing (equivalent to RMA)
    atr_period = period
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # highest high and lowest low over period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(period-1, len(high)):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    # chop calculation
    chop = np.zeros_like(close)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            log_sum = np.log10(np.sum(atr[i-period+1:i+1]) / (highest_high[i] - lowest_low[i]))
            chop[i] = 100 * log_sum / np.log10(period)
        else:
            chop[i] = 50  # neutral when no range
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend direction
    kama_1w = calculate_kama(df_1w['close'].values)
    kama_1w_slope = np.diff(kama_1w, prepend=kama_1w[0])  # positive = uptrend
    kama_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_slope)
    
    # Calculate daily Choppiness Index for regime filter
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate RSI on 12h chart
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    rsi_period = 14
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    for i in range(rsi_period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period to prevent churn
    
    start_idx = max(50, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_slope_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Trend filter: weekly KAMA slope direction
        kama_uptrend = kama_1w_slope_aligned[i] > 0
        kama_downtrend = kama_1w_slope_aligned[i] < 0
        
        # Chop filter: choppy market (good for mean reversion)
        choppy = chop_1d_aligned[i] > 61.8
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        if position == 0:
            # Require minimum 2 bars since last exit to prevent churn
            if bars_since_entry >= 2:
                # Long entry: uptrend, oversold, choppy market
                if kama_uptrend and rsi_oversold and choppy:
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short entry: downtrend, overbought, choppy market
                elif kama_downtrend and rsi_overbought and choppy:
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Long exit: RSI returns to neutral or trend changes to downtrend
            if rsi_neutral[i] or not kama_uptrend:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                bars_since_entry += 1
        elif position == -1:
            # Short exit: RSI returns to neutral or trend changes to uptrend
            if rsi_neutral[i] or not kama_downtrend:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                bars_since_entry += 1
    
    return signals