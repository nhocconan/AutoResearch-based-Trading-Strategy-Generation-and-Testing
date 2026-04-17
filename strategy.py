#!/usr/bin/env python3
"""
1h_RSI_Pullback_TrendFollow
Hypothesis: In 1h timeframe, buy pullbacks to EMA21 during strong 4h uptrend (EMA50 > EMA200) and sell rallies to EMA21 during strong 4h downtrend (EMA50 < EMA200). Uses 4h trend for direction, 1h RSI for entry timing, and session filter (08-20 UTC) to reduce noise. Target: 20-40 trades/year for low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Data (HTF for trend direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 and EMA200 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h indicators
    close_s = pd.Series(close)
    ema21_1h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    rsi_1h = calculate_rsi(close, 14)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i]) or
            np.isnan(ema21_1h[i]) or
            np.isnan(rsi_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine 4h trend
        uptrend = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        downtrend = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: uptrend + pullback to EMA21 (RSI < 40)
            if uptrend and close[i] <= ema21_1h[i] and rsi_1h[i] < 40:
                signals[i] = 0.20
                position = 1
                continue
            # Short: downtrend + rally to EMA21 (RSI > 60)
            elif downtrend and close[i] >= ema21_1h[i] and rsi_1h[i] > 60:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when RSI > 70 (overbought) or close below EMA21
            if rsi_1h[i] > 70 or close[i] < ema21_1h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when RSI < 30 (oversold) or close above EMA21
            if rsi_1h[i] < 30 or close[i] > ema21_1h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper Wilder smoothing"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # Wilder smoothing
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    # Set first period values to NaN
    rsi[:period] = np.nan
    
    return rsi

name = "1h_RSI_Pullback_TrendFollow"
timeframe = "1h"
leverage = 1.0