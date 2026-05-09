#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Momentum_1dTrend_Filter
Hypothesis: KAMA identifies trend direction with adaptive smoothing, RSI provides momentum confirmation, and 1d trend filter (price > EMA200) ensures alignment with higher timeframe trend. Designed for low trade frequency (12-37/year) to minimize fee drift. Works in both bull and bear markets by following the dominant trend on higher timeframe.
"""

name = "12h_KAMA_Direction_RSI_Momentum_1dTrend_Filter"
timeframe = "12h"
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
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    def calculate_kama(close, er_length=10, fast=2, slow=30):
        kama = np.full_like(close, np.nan)
        if len(close) < er_length + 1:
            return kama
        
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        rsi = np.full_like(close, np.nan)
        if len(close) < length + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (ema_200_1d[i-1] * 199 + close_1d[i]) / 200
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate KAMA and RSI on 12h data
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(30, 20)  # Ensure KAMA, RSI, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price > KAMA (uptrend) AND RSI > 50 (bullish momentum) AND price > EMA200_1d (uptrend on 1d) AND volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price < KAMA (downtrend) AND RSI < 50 (bearish momentum) AND price < EMA200_1d (downtrend on 1d) AND volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price < KAMA OR RSI < 40 (loss of momentum)
                if close[i] < kama[i] or rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price > KAMA OR RSI > 60 (loss of momentum)
                if close[i] > kama[i] or rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals