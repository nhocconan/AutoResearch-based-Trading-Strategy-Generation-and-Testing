#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Trend_Filter
Hypothesis: 4h KAMA trend direction with RSI momentum and volume confirmation
KAMA adapts to market noise, reducing whipsaws in ranging markets
RSI > 50 for long, < 50 for short ensures momentum alignment
Volume > 1.5x 20-period average confirms institutional participation
Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years)
Works in bull/bear via adaptive trend filter and momentum confirmation
"""

name = "4h_1d_KAMA_RSI_Trend_Filter"
timeframe = "4h"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - trend identification
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_length))
        abs_change = np.abs(np.diff(close, prepend=close[0]))
        er_num = change
        er_den = np.sum(np.lib.stride_tricks.sliding_window_view(abs_change, er_length), axis=-1)
        # Pad the denominator to match length
        er_den_padded = np.full_like(close, np.nan)
        er_den_padded[er_length-1:] = er_den
        er = np.where(er_den_padded > 0, er_num / er_den_padded, 0)
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
        return kama
    
    # RSI (Relative Strength Index)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        # First average
        if len(close) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # 4h data for KAMA and RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for indicator calculation
        return np.zeros(n)
    
    # Calculate KAMA and RSI on 4h data
    kama_4h = calculate_kama(df_4h['close'].values, 10, 2, 30)
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    
    # Align indicators to lower timeframe (if needed, though same TF here)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d data for trend context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Simple 1d EMA for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend: price above KAMA = uptrend, below = downtrend
        kama_trend_up = close[i] > kama_4h_aligned[i]
        kama_trend_down = close[i] < kama_4h_aligned[i]
        
        # RSI momentum: > 50 = bullish momentum, < 50 = bearish momentum
        rsi_bullish = rsi_4h_aligned[i] > 50
        rsi_bearish = rsi_4h_aligned[i] < 50
        
        if position == 0:
            # Long: price above KAMA (uptrend) AND RSI > 50 (bullish momentum) AND volume confirmation
            if (kama_trend_up and rsi_bullish and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) AND RSI < 50 (bearish momentum) AND volume confirmation
            elif (kama_trend_down and rsi_bearish and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA OR RSI < 40 (losing momentum)
            if (not kama_trend_up) or (rsi_4h_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA OR RSI > 60 (losing momentum)
            if (not kama_trend_down) or (rsi_4h_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals