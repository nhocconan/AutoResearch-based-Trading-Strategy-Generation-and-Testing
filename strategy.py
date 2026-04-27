#!/usr/bin/env python3
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
    
    # Get 1d data for RSI and moving averages
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 200-day EMA
    ema_period = 200
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_200_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                            ema_200_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema_period_w = 50
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period_w:
        ema_50_1w[ema_period_w - 1] = np.mean(close_1w[:ema_period_w])
        for i in range(ema_period_w, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * (2 / (ema_period_w + 1)) + 
                           ema_50_1w[i-1] * (1 - (2 / (ema_period_w + 1))))
    
    # Align indicators to daily timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need RSI, EMA200, weekly EMA50, and volume MA
    start_idx = max(14, 200, 50, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: RSI < 35 (oversold) + price > 200EMA + weekly EMA50 rising + volume spike
            if (rsi_1d_aligned[i] < 35 and 
                price > ema_200_1d_aligned[i] and
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: RSI > 65 (overbought) + price < 200EMA + weekly EMA50 falling + volume spike
            elif (rsi_1d_aligned[i] > 65 and 
                  price < ema_200_1d_aligned[i] and
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion) or weekly EMA50 turns down
            if (rsi_1d_aligned[i] > 50 or 
                ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion) or weekly EMA50 turns up
            if (rsi_1d_aligned[i] < 50 or 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_MeanReversion_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0