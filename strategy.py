#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(14) + 1w trend filter + volume confirmation
# Uses adaptive KAMA to follow trend with less whipsaw, RSI for overbought/oversold entries,
# weekly EMA20 for trend filter, and volume spike for momentum confirmation.
# Designed to work in both bull and bear markets by adapting to trend changes.
# Target: 8-20 trades/year (32-80 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close(t) - close(t-1)| over 10 periods
    # Fix dimensions: volatility needs same length as change
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+10], n=1))) if i+10 <= len(close) else np.nan 
                          for i in range(len(close))])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start after KAMA warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA (uptrend), RSI < 30 (oversold), above weekly EMA20, volume spike
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA (downtrend), RSI > 70 (overbought), below weekly EMA20, volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signal or trend change
            if position == 1:
                # Exit long: price < KAMA or RSI > 70
                if close[i] < kama[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price > KAMA or RSI < 30
                if close[i] > kama[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_KAMA_RSI_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0