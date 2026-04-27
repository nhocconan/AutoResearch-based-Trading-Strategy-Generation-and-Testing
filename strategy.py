#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# In trending markets, price pulls back to EMA21 and resumes trend with volume surge.
# Uses 4h EMA50 for trend direction and volume spike for confirmation.
# Designed to work in both bull (long on pullbacks) and bear (short on rallies) markets.
# Target: 20-40 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema_50_4h = np.full(len(df_4h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i < 49:
            ema_50_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_50_4h[i-1]):
                ema_50_4h[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_50_4h[i] = close_4h[i] * alpha + ema_50_4h[i-1] * (1 - alpha)
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # 1h EMA21 for pullback entries
    ema_21 = np.full(n, np.nan)
    alpha21 = 2 / (21 + 1)
    for i in range(n):
        if i < 20:
            ema_21[i] = np.mean(close[:i+1]) if i > 0 else close[i]
        else:
            if np.isnan(ema_21[i-1]):
                ema_21[i] = np.mean(close[i-20:i+1])
            else:
                ema_21[i] = close[i] * alpha21 + ema_21[i-1] * (1 - alpha21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_4h_aligned[i-1]):
            trend_up = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            trend_down = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: pullback to EMA21 in uptrend + volume spike
            if (close[i] <= ema_21[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: rally to EMA21 in downtrend + volume spike
            elif (close[i] >= ema_21[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend turns down or price breaks above EMA21 with momentum
            if (not trend_down or 
                close[i] > ema_21[i] * 1.01):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend turns up or price breaks below EMA21 with momentum
            if (not trend_up or 
                close[i] < ema_21[i] * 0.99):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_Pullback_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0