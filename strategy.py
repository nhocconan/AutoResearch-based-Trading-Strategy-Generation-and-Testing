#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA with 1-day RSI filter and volume confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# Combined with daily RSI extremes and volume spikes, it captures strong trends while avoiding whipsaws.
# Target: 15-30 trades per year (60-120 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14) for 1-day timeframe
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros_like(close_1d)
    rs[13:] = avg_gain[13:] / np.where(avg_loss[13:] == 0, 1e-10, avg_loss[13:])
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA on 12h timeframe
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) >= length else np.array([0])
        # Handle array operations properly
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+length]))) if i+length <= len(close) else 0 
                              for i in range(len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constants
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_12h = kama(close, length=10, fast=2, slow=30)
    
    # Average volume (24-period = 12 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_12h[i]
        rsi_val = rsi_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Price above KAMA + RSI < 30 (oversold) + volume confirmation
            if (price > kama_val and
                rsi_val < 30 and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price below KAMA + RSI > 70 (overbought) + volume confirmation
            elif (price < kama_val and
                  rsi_val > 70 and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below KAMA or RSI > 70
            if (price < kama_val or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above KAMA or RSI < 30
            if (price > kama_val or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0