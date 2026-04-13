#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with RSI momentum filter and volume confirmation.
# KAMA adapts to market efficiency, reducing whipsaws in ranging markets.
# RSI ensures momentum alignment with trend direction.
# Volume confirmation filters low-conviction moves.
# Designed for 12h timeframe to capture multi-day trends in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) - 12h
    def kama(price, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=10))  # 10-period net change
        volatility = np.sum(np.abs(np.diff(price)), axis=1)  # 10-period volatility
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(price, np.nan)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, er_length=10, fast=2, slow=30)
    
    # Calculate RSI (14-period) - 12h
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, period=14)
    
    # Calculate average volume (20-period) - 12h
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align daily RSI for trend filter
    close_1d = df_1d['close'].values
    def rsi_1d(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d_vals = rsi_1d(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_vals)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama_vals[i]
        rsi_val = rsi_vals[i]
        daily_rsi = rsi_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 + daily RSI > 50 + volume confirmation
            if (price > kama_val and 
                rsi_val > 50 and 
                daily_rsi > 50 and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price below KAMA + RSI < 50 + daily RSI < 50 + volume confirmation
            elif (price < kama_val and 
                  rsi_val < 50 and 
                  daily_rsi < 50 and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA OR RSI < 40
            if (price < kama_val or 
                rsi_val < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above KAMA OR RSI > 60
            if (price > kama_val or 
                rsi_val > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0