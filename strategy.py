#!/usr/bin/env python3
# [24920] 4h_1d_momentum_pullback_v1
# Hypothesis: 4-hour momentum pullback with 1-day trend filter. Long when RSI(14) crosses above 50 on pullback to EMA21 with bullish 1-day EMA50. Short when RSI(14) crosses below 50 on pullback to EMA21 with bearish 1-day EMA50. Uses RSI for momentum entry and EMA21 for dynamic support/resistance. Designed to capture swings in both bull and bear markets by trading with the higher timeframe trend during pullbacks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_momentum_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate 4-hour EMA21 for dynamic support/resistance
    ema_21 = np.full(n, np.nan)
    if len(close) >= 21:
        alpha = 2.0 / (21 + 1)
        ema_21[20] = np.mean(close[:21])
        for i in range(21, n):
            ema_21[i] = alpha * close[i] + (1 - alpha) * ema_21[i-1]
    
    # Calculate 4-hour RSI(14)
    rsi = np.full(n, np.nan)
    if len(close) >= 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
    
    # Align 1-day EMA50 to 4-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_21[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        ema21_val = ema_21[i]
        rsi_val = rsi[i]
        trend_up_1d = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI crosses below 50 or price breaks below EMA21
            if rsi_val < 50 or price < ema21_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI crosses above 50 or price breaks above EMA21
            if rsi_val > 50 or price > ema21_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI crosses above 50 on pullback to EMA21 with bullish 1D trend
            if (rsi_val > 50 and rsi[i-1] <= 50 and 
                price >= ema21_val * 0.995 and price <= ema21_val * 1.005 and  # Near EMA21
                trend_up_1d):
                position = 1
                signals[i] = 0.25
            # Enter short: RSI crosses below 50 on pullback to EMA21 with bearish 1D trend
            elif (rsi_val < 50 and rsi[i-1] >= 50 and 
                  price >= ema21_val * 0.995 and price <= ema21_val * 1.005 and  # Near EMA21
                  not trend_up_1d):
                position = -1
                signals[i] = -0.25
    
    return signals