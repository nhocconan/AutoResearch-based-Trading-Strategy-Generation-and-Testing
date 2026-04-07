#!/usr/bin/env python3
"""
12h_kama_rsi_1d_trend_volume_v1
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with RSI for momentum confirmation and volume for conviction, this strategy captures sustained moves.
The 1d trend filter ensures alignment with higher timeframe momentum, reducing false signals.
Targeting 12-37 trades/year by requiring KAMA trend alignment, RSI momentum, and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - 12h
    def kama(price, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = np.power(er * (fast_sc/(slow_sc - fast_sc)) + (1 - er), 2)
        # Initialize KAMA
        kama_vals = np.full_like(price, np.nan, dtype=np.float64)
        kama_vals[0] = price[0]
        for i in range(1, len(price)):
            if np.isnan(kama_vals[i-1]):
                kama_vals[i] = price[i]
            else:
                kama_vals[i] = kama_vals[i-1] + sc[i] * (price[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close)
    
    # RSI (14) - 12h
    def rsi(price, period=14):
        delta = np.diff(price, prepend=price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Warmup for KAMA/RSI
        # Skip if required data not available
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI turns weak (<40) OR trend turns down
            if (close[i] < kama_vals[i] or 
                rsi_vals[i] < 40 or 
                close[i] < ema50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI turns weak (>60) OR trend turns up
            if (close[i] > kama_vals[i] or 
                rsi_vals[i] > 60 or 
                close[i] > ema50_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above KAMA + RSI > 50 (bullish momentum) + volume + uptrend
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] > 50 and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA + RSI < 50 (bearish momentum) + volume + downtrend
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] < 50 and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals