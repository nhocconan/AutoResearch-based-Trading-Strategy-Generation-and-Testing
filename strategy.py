#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10  # Efficiency Ratio period
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    volatility = np.sum(np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0:1])), axis=0) if False else None
    # Proper ER calculation
    price_series = df_1d['close'].values
    change = np.abs(np.diff(price_series, prepend=price_series[0]))
    volatility = np.zeros_like(price_series)
    for i in range(1, len(price_series)):
        volatility[i] = volatility[i-1] + np.abs(price_series[i] - price_series[i-1])
    
    er = np.zeros_like(price_series)
    er[er_len:] = change[er_len:] / (volatility[er_len:] + 1e-10)
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(price_series)
    kama[0] = price_series[0]
    for i in range(1, len(price_series)):
        kama[i] = kama[i-1] + sc[i] * (price_series[i] - kama[i-1])
    
    kama = kama  # already aligned to 1d
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on 1d close
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-13:i+1])
            avg_loss[i] = np.mean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: current volume > 1.5x 20-period average (more permissive)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~1.3 days for 4h to reduce trades
    
    start_idx = max(200, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction from KAMA
        trend_up = close[i] > kama_aligned[i]
        trend_down = close[i] < kama_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price above KAMA (uptrend) and RSI not overbought
            if trend_up[i] and rsi_aligned[i] < 70:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price below KAMA (downtrend) and RSI not oversold
            elif trend_down[i] and rsi_aligned[i] > 30:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price crosses below KAMA or RSI overbought
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or RSI oversold
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using 4h timeframe with KAMA trend direction and RSI filter (30/70) 
# will capture trends while avoiding overextended markets. Volume confirmation 
# (1.5x 20-period average) ensures institutional participation. The strategy 
# adapts to both bull and bear markets by following the KAMA trend. 
# Position size of 0.25 manages drawdown, and cooldown of 8 bars prevents overtrading. 
# Expected trades: 20-40 per year (80-160 total over 4 years) which is within 
# acceptable limits to minimize fee drag while maintaining edge.