#!/usr/bin/env python3
# 4h_RSI_MeanReversion_CamarillaPivot
# Hypothesis: Mean reversion at daily Camarilla pivot support/resistance with RSI confirmation. Goes long when price touches S1 and RSI < 30 (oversold), short when price touches R1 and RSI > 70 (overbought). Exits on opposite pivot level or RSI mean reversion (RSI > 50 for longs, < 50 for shorts). Designed to work in both bull and bear markets by exploiting intraday reversals at institutional levels. Targets 20-40 trades/year via strict price-level + RSI confluence.

name = "4h_RSI_MeanReversion_CamarillaPivot"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_ = high - low
    r1 = close + (range_ * 1.1 / 12)
    s1 = close - (range_ * 1.1 / 12)
    return r1, s1, pivot

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder's smoothing"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing: first average is simple average
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Subsequent values: smoothed average
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Set first 'period' values to 50 (neutral) as warmup
    rsi[:period] = 50
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Daily Camarilla Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r1_1d, s1_1d, pivot_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Camarilla to 4h
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- RSI (14) on 4h closes ---
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(rsi[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below S1 + RSI oversold (<30)
            if (close[i] <= s1_4h[i] and 
                rsi[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 + RSI overbought (>70)
            elif (close[i] >= r1_4h[i] and 
                  rsi[i] > 70):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price at or above R1 OR RSI > 50 (mean reversion)
                if (close[i] >= r1_4h[i] and close[i-1] < r1_4h[i-1]) or \
                   (rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price at or below S1 OR RSI < 50 (mean reversion)
                if (close[i] <= s1_4h[i] and close[i-1] > s1_4h[i-1]) or \
                   (rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals