#!/usr/bin/env python3
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
    
    # Get daily data for 200-day EMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200-day EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200 = np.zeros(len(close_1d))
    ema_200[0] = close_1d[0]
    alpha = 2 / (200 + 1)
    for i in range(1, len(close_1d)):
        ema_200[i] = alpha * close_1d[i] + (1 - alpha) * ema_200[i-1]
    
    # Align daily EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 4-period RSI for entry signal
    def calculate_rsi(prices, period):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4 = calculate_rsi(close, 4)
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need EMA200 and RSI4
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(rsi_4[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # RSI condition: RSI(4) < 30 for long, > 70 for short
        rsi_long = rsi_4[i] < 30
        rsi_short = rsi_4[i] > 70
        
        # Trend filter: price above/below daily EMA200
        above_ema200 = price > ema_200_aligned[i]
        below_ema200 = price < ema_200_aligned[i]
        
        if position == 0:
            # Long: oversold RSI + above EMA200 + volume
            if rsi_long and above_ema200 and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI + below EMA200 + volume
            elif rsi_short and below_ema200 and volume_confirmation:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral or trend changes
            if rsi_4[i] >= 50 or price < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or trend changes
            if rsi_4[i] <= 50 or price > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1D_EMA200_RSI4_Volume"
timeframe = "4h"
leverage = 1.0