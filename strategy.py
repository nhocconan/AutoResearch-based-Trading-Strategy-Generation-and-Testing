#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + 1d RSI mean reversion + volume confirmation
# Uses 12h Kaufman Adaptive Moving Average for trend direction (long when price > KAMA, short when price < KAMA)
# and 1d RSI for mean reversion entries (long when RSI < 30, short when RSI > 70)
# Volume > 1.5x 20-period average confirms momentum. Trend filter avoids counter-trend trades.
# Target: 15-25 trades/year to minimize fee decay while capturing strong momentum with mean reversion entries.
# Focus on BTC/ETH as primary assets with proven KAMA and RSI edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h KAMA for trend filter
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    # Handle volatility calculation for ER
    volatility_series = []
    for i in range(len(close_12h)):
        if i < 10:
            volatility_series.append(np.nan)
        else:
            vol_sum = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
            volatility_series.append(vol_sum if vol_sum > 0 else 1)
    volatility = np.array(volatility_series)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Calculate average gain and loss
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # Initial average gain/loss
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        # Wilder's smoothing
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(20, vol_period, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 12h KAMA
        uptrend = price > kama_12h_aligned[i]
        downtrend = price < kama_12h_aligned[i]
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) AND RSI < 30 (oversold) with volume confirmation
            if uptrend and rsi_14_1d_aligned[i] < 30 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price below KAMA (downtrend) AND RSI > 70 (overbought) with volume confirmation
            elif downtrend and rsi_14_1d_aligned[i] > 70 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below KAMA or RSI > 70 (overbought)
            if price < kama_12h_aligned[i] or rsi_14_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price above KAMA or RSI < 30 (oversold)
            if price > kama_12h_aligned[i] or rsi_14_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0