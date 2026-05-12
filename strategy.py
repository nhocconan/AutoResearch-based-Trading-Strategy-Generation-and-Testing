#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Filter_Volume
# Hypothesis: Use KAMA trend direction on 1d combined with RSI(14) mean reversion and volume confirmation.
# Enter long when KAMA trend is up, RSI < 30 (oversold), and volume > 20-period average.
# Enter short when KAMA trend is down, RSI > 70 (overbought), and volume > 20-period average.
# Exit when RSI crosses back to neutral (40-60 range) or trend fails.
# Designed for low frequency (7-25 trades/year) to avoid fee drag. Works in bull (buy dips in uptrend)
# and bear (sell rallies in downtrend) with trend filter and mean reversion.

name = "1d_KAMA_Trend_RSI_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_fast=2, er_slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Returns KAMA array.
    """
    n = len(close)
    kama_arr = np.zeros(n)
    er = np.zeros(n)
    sc = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if n > 1 else np.zeros(n-1)
    # Simplified volatility calculation for efficiency
    vol = np.zeros(n)
    for i in range(1, n):
        vol[i] = vol[i-1] + np.abs(close[i] - close[i-1])
        if i >= 10:
            vol[i] -= np.abs(close[i-10] - close[i-11])
    
    for i in range(n):
        if i >= 10:
            ch = np.abs(close[i] - close[i-10])
            vol_sum = vol[i] - (vol[i-10] if i >= 10 else 0)
            er[i] = ch / vol_sum if vol_sum != 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2 / (er_fast + 1)
    slow_sc = 2 / (er_slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama_arr[0] = close[0]
    for i in range(1, n):
        kama_arr[i] = kama_arr[i-1] + sc[i] * (close[i] - kama_arr[i-1])
    
    return kama_arr

def rsi(close, period=14):
    """
    Relative Strength Index (RSI).
    Returns RSI array.
    """
    n = len(close)
    rsi_arr = np.full(n, 50.0)  # Initialize to neutral
    
    if n < period + 1:
        return rsi_arr
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initial average
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # Wilder's smoothing
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    # Calculate RSI
    for i in range(period+1, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi_arr[i] = 100 - (100 / (1 + rs))
        else:
            rsi_arr[i] = 100
    
    return rsi_arr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (more stable than daily for long-term trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on weekly data for trend filter
    kama_1w = kama(close_1w, er_fast=2, er_slow=30)
    
    # Calculate RSI on daily price
    rsi_14 = rsi(close, period=14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure indicators are stable (14 RSI + 20 vol + buffer)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly KAMA
        trend_up = close[i] > kama_1w_aligned[i]
        trend_down = close[i] < kama_1w_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # RSI conditions
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_neutral = (rsi_14[i] >= 40) and (rsi_14[i] <= 60)
        
        if position == 0:
            # LONG: KAMA trend up, RSI oversold, volume confirmation
            if trend_up and rsi_oversold and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trend down, RSI overbought, volume confirmation
            elif trend_down and rsi_overbought and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI returns to neutral or trend fails
            if rsi_neutral or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral or trend fails
            if rsi_neutral or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals