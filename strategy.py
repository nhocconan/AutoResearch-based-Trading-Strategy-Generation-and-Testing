#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA filter and volume confirmation
# - Uses Kaufman Adaptive Moving Average (KAMA) on 1d to capture trend direction
# - 1w EMA200 as higher-timeframe trend filter to avoid counter-trend trades
# - Volume spike (2x 20-day average) confirms momentum
# - Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe
# - Works in bull/bear by using 1w trend filter and KAMA's adaptive nature

name = "1d_KAMA_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER period=10, Fast SC=2/(2+1), Slow SC=2/(30+1)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # For simplicity, use rolling volatility and directional change
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        directional_change = np.abs(close_1d[i] - close_1d[i-10])
        total_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        if total_change > 0:
            er[i] = directional_change / total_change
        else:
            er[i] = 0
    # Smooth ER
    for i in range(1, len(er)):
        er[i] = 0.9 * er[i-1] + 0.1 * er[i]
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (itself)
    kama_aligned = kama  # Already on 1d
    
    # 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 200)  # Ensure enough data for KAMA and EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA and 1w EMA200 rising + volume spike
            long_cond = (close[i] > kama_aligned[i] and 
                        ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price below KAMA and 1w EMA200 falling + volume spike
            short_cond = (close[i] < kama_aligned[i] and 
                         ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals