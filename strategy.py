#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Range_200MA_V1
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI range filtering and 200-period moving average to avoid whipsaws.
Designed for low trade frequency (20-50 trades/year) with strong trend confirmation
to work in both bull and bear markets by filtering counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    def kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate 200-period SMA for long-term trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Calculate RSI (14-period)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, 14)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_avg_1d)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate indicators
    kama_vals = kama(close, 10, 2, 30)
    sma200_vals = sma200
    rsi_vals = rsi_vals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 200  # SMA200 needs 200 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(sma200_vals[i]) or 
            np.isnan(rsi_vals[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_vals[i]
        sma200_val = sma200_vals[i]
        rsi_val = rsi_vals[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price above KAMA (uptrend), above SMA200, RSI not overbought, volume spike
            if (close_val > kama_val and 
                close_val > sma200_val and 
                rsi_val < 70 and 
                vol_spike):
                signals[i] = size
                position = 1
            # Short conditions: price below KAMA (downtrend), below SMA200, RSI not oversold, volume spike
            elif (close_val < kama_val and 
                  close_val < sma200_val and 
                  rsi_val > 30 and 
                  vol_spike):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if close_val < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if close_val > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_Range_200MA_V1"
timeframe = "4h"
leverage = 1.0