#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend filter with 1w RSI extremes and volume confirmation.
# KAMA adapts to market conditions - fast in trends, slow in ranges.
# Weekly RSI >70 or <30 identifies extremes in the longer trend.
# Volume > 1.5x 20-day average confirms institutional participation.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
name = "1d_KAMA_RSI_Extremes_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d KAMA trend filter
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1w RSI for extreme conditions
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    rsi_period = 14
    delta = np.diff(df_1w['close'].values, prepend=df_1w['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1w RSI to 1d
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Volume confirmation: volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) + weekly RSI < 30 (oversold) + volume confirmation
            if (price > kama[i] and rsi_1d[i] < 30 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + weekly RSI > 70 (overbought) + volume confirmation
            elif (price < kama[i] and rsi_1d[i] > 70 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA (trend change) or weekly RSI > 70 (overbought)
            if price < kama[i] or rsi_1d[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA (trend change) or weekly RSI < 30 (oversold)
            if price > kama[i] or rsi_1d[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals