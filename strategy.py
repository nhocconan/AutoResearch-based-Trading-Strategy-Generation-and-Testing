#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in sideways markets.
# RSI filters overextended entries. Works in both bull and bear by following the adaptive trend.
# Uses 1d trend filter to avoid counter-trend trades. Volume spike confirms momentum.
# Designed for low trade frequency (<50/year) to minimize fee drag.

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive trend)
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, n=length))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.zeros_like(close_prices)
        er[length:] = change[length-1:] / np.maximum(volatility[length-1:], 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.full_like(close_prices, np.nan)
        kama[length] = close_prices[length]
        for i in range(length+1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= length:
            avg_gain[length-1] = np.mean(gain[:length])
            avg_loss[length-1] = np.mean(loss[:length])
            
            for i in range(length, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate KAMA on 4h data
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI on 4h data
    rsi = calculate_rsi(close, length=14)
    
    # Volume spike filter
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure KAMA, RSI, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA (uptrend), RSI < 70 (not overbought), 
            # price > 1d EMA50 (1d uptrend), volume spike
            if (close[i] > kama[i] and 
                rsi[i] < 70 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend), RSI > 30 (not oversold),
            # price < 1d EMA50 (1d downtrend), volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] > 30 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 70 (overbought) OR trend reversal
            if (close[i] < kama[i] or 
                rsi[i] > 70 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 30 (oversold) OR trend reversal
            if (close[i] > kama[i] or 
                rsi[i] < 30 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals