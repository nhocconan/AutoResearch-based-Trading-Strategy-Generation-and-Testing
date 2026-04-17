#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # === Weekly RSI (14-period) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate RSI using Wilder's smoothing
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing with proper seeding
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[avg_loss == 0] = 100
    
    # === Daily Close for Price Action ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Daily EMA (21-period) ===
    ema_21_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            ema_21_1d[i] = close_1d[0]
        else:
            ema_21_1d[i] = (close_1d[i] * 0.0909) + (ema_21_1d[i-1] * 0.9091)
    
    # === Align indicators to daily timeframe ===
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Weekly RSI < 30 (oversold) + Daily price > EMA21
            if (rsi_1w_aligned[i] < 30 and 
                close[i] > ema_21_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Weekly RSI > 70 (overbought) + Daily price < EMA21
            elif (rsi_1w_aligned[i] > 70 and 
                  close[i] < ema_21_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Weekly RSI crosses above 50
            if rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly RSI crosses below 50
            if rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRSI_EMA21_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0