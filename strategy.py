#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with 1-week RSI filter and volume confirmation
# Uses KAMA's adaptive smoothing to reduce whipsaw in choppy markets
# RSI filter avoids overbought/oversold extremes, volume confirms conviction
# Designed for low trade frequency (target: 50-100 total over 4 years) to minimize fee drag
# Works in bull/bear via trend filter and volatility-adjusted position sizing

name = "daily_kama_rsi_vol_v1"
timeframe = "1d"
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
    
    # 1-week data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility[0] = np.abs(close[0] - close[0])  # Will be updated properly below
    
    # Recalculate volatility properly (10-period sum of absolute changes)
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    volatility[:10] = volatility[10] if len(close) > 10 else 0
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = change / volatility
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on weekly
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(np.isnan(rsi_1w), 50, rsi_1w)  # Handle initial values
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA warmup
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Long: price above KAMA, RSI < 50 (not overbought), volume confirmation
            if (close[i] > kama[i] and 
                rsi_1w_aligned[i] < 50 and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI > 50 (not oversold), volume confirmation
            elif (close[i] < kama[i] and 
                  rsi_1w_aligned[i] > 50 and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
    
    return signals