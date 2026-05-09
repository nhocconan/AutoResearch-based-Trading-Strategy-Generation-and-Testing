#!/usr/bin/env python3
# 1D_KAMA_Trend_Stochastic_RSI_Confirmation
# Hypothesis: On 1d timeframe, enter long when KAMA indicates uptrend, Stochastic RSI shows oversold conditions, and weekly trend is up. Short when KAMA indicates downtrend, Stochastic RSI shows overbought conditions, and weekly trend is down. Weekly trend filter avoids counter-trend trades, reducing whipsaw in bear markets. KAMA provides adaptive trend following, while Stochastic RSI provides mean-reversion entry points. Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "1D_KAMA_Trend_Stochastic_RSI_Confirmation"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter (price above/below weekly EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA 20 for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = close_1w > ema_20_1w  # uptrend if price above weekly EMA20
    
    # Get daily data for KAMA and Stochastic RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    volatility = np.append(np.abs(close_1d[0] - close_1d[0]), volatility[1:])  # reset first
    # Actually, volatility is sum of absolute changes over period
    volatility = np.zeros_like(close_1d)
    for i in range(len(volatility)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    for i in range(len(er)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA 2
    slow_sc = 2 / (30 + 1)  # EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_trend_up = close_1d > kama  # price above KAMA indicates uptrend
    
    # Stochastic RSI calculation
    # First, calculate RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi = np.zeros_like(close_1d)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Initial average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        # Calculate RSI
        for i in range(14, len(rsi)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
    
    # Stochastic of RSI
    stoch_rsi = np.zeros_like(rsi)
    lookback = 14
    for i in range(lookback, len(rsi)):
        rsi_window = rsi[i-lookback+1:i+1]
        min_rsi = np.min(rsi_window)
        max_rsi = np.max(rsi_window)
        if max_rsi - min_rsi > 0:
            stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
        else:
            stoch_rsi[i] = 50  # neutral if no range
    
    # Stochastic RSI signals
    stoch_rsi_oversold = stoch_rsi < 20
    stoch_rsi_overbought = stoch_rsi > 80
    
    # Align weekly trend and KAMA trend to daily
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    kama_trend_up_aligned = align_htf_to_ltf(prices, df_1d, kama_trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_trend_up_aligned[i]) or np.isnan(kama_trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA uptrend + Stochastic RSI oversold + weekly uptrend
            if kama_trend_up_aligned[i] and stoch_rsi_oversold[i] and weekly_trend_up_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downtrend + Stochastic RSI overbought + weekly downtrend
            elif not kama_trend_up_aligned[i] and stoch_rsi_overbought[i] and not weekly_trend_up_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA downtrend or Stochastic RSI overbought
            if not kama_trend_up_aligned[i] or stoch_rsi_overbought[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA uptrend or Stochastic RSI oversold
            if kama_trend_up_aligned[i] or stoch_rsi_oversold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals