#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: KAMA adapts to market noise, providing a smooth trend filter. Combined with RSI momentum and volume confirmation on daily timeframe, it captures sustained moves while avoiding whipsaws. Weekly trend filter ensures alignment with higher timeframe momentum. Designed for 1-2 trades per month, targeting 12-24 trades/year to minimize fee drag and work in both bull and bear markets via adaptive trend strength.
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate indicators
    kama_val = kama(close, er_len=10, fast=2, slow=30)
    rsi_val = rsi(close, length=14)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_now = kama_val[i]
        rsi_now = rsi_val[i]
        ema_trend = ema50_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, and weekly uptrend with volume spike
            if close[i] > kama_now and rsi_now > 50 and close[i] > ema_trend and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price below KAMA, RSI < 50, and weekly downtrend with volume spike
            elif close[i] < kama_now and rsi_now < 50 and close[i] < ema_trend and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below KAMA or RSI < 40 or weekly trend turns down
            if close[i] < kama_now or rsi_now < 40 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above KAMA or RSI > 60 or weekly trend turns up
            if close[i] > kama_now or rsi_now > 60 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
leverage = 1.0