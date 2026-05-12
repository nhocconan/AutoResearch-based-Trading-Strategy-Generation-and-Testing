#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Filter
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) for trend direction on 12h,
# filtered by RSI(14) to avoid overbought/oversold extremes and volume confirmation.
# Enter long when price > KAMA and RSI < 50, short when price < KAMA and RSI > 50.
# Exit when price crosses back through KAMA. Designed for low frequency (10-30 trades/year)
# to avoid fee drag. Works in bull (ride trends) and bear (catch reversals from extremes).

name = "12h_KAMA_Trend_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Returns KAMA array.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=period))  # |close[t] - close[t-period]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over period
    
    # Handle first period elements
    for i in range(period, n):
        if volatility[i] > 0:
            er = change[i] / volatility[i]
        else:
            er = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    Returns RSI array.
    """
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    # Price changes
    delta = np.diff(close)
    
    # Gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Average gain and loss
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    for i in range(n):
        if i < period:
            if i > 0:
                avg_gain[i] = np.mean(gain[:i]) if i > 0 else 0
                avg_loss[i] = np.mean(loss[:i]) if i > 0 else 0
            else:
                avg_gain[i] = 0
                avg_loss[i] = 0
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    # Calculate RSI
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for additional filters (optional, can be removed if not needed)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI on 12h data
    rsi = calculate_rsi(close, period=14)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable (max of KAMA period, RSI period, vol MA)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI filter: avoid extremes (RSI < 50 for long, RSI > 50 for short)
        rsi_ok_long = rsi[i] < 50
        rsi_ok_short = rsi[i] > 50
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: price > KAMA, RSI < 50, volume confirmation
            if price_above_kama and rsi_ok_long and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA, RSI > 50, volume confirmation
            elif price_below_kama and rsi_ok_short and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals