#!/usr/bin/env python3
# 1D_KAMA_Trend_With_RSI_Filter_and_Volume
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# Combined with RSI(14) for momentum confirmation and volume spike for institutional validation.
# Uses weekly trend filter to avoid counter-trend trades. Designed for low turnover (target: 15-25 trades/year).

name = "1D_KAMA_Trend_With_RSI_Filter_and_Volume"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.concatenate([[0], volatility[:-1]])  # sum of abs changes over er_length
        
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[0:period])
            avg_loss[period-1] = np.mean(loss[0:period])
        
        # Wilder smoothing
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema_21_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[0:21])
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = (ema_21_1w[i-1] * 20 + close_1w[i]) / 21
    
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21, 14)  # Ensure volume MA, weekly EMA, and RSI are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend) AND RSI > 50 (bullish momentum) 
            # AND volume spike AND weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                volume_ratio[i] > 1.5 and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend) AND RSI < 50 (bearish momentum)
            # AND volume spike AND weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  volume_ratio[i] > 1.5 and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 40 (losing momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 60 (losing bearish momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals