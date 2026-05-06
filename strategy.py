#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily KAMA trend direction with 12h EMA filter and volume spike
# - Uses 1d KAMA (adaptive moving average) to identify trend direction
# - Uses 12h EMA50 as higher timeframe trend confirmation
# - Uses 4h volume spike for entry confirmation
# - Enters long when price > 1d KAMA AND price > 12h EMA50 with volume spike
# - Enters short when price < 1d KAMA AND price < 12h EMA50 with volume spike
# - Exits when price crosses back below/above 1d KAMA
# - Designed to capture trends with adaptive trend filter and avoid whipsaws
# - Target: 80-180 total trades over 4 years (20-45/year) with 0.25 position sizing

name = "4h_1dKAMA_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_fast=2, er_slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA
    close_1d = df_1d['close'].values
    kama_1d = calculate_kama(close_1d, er_fast=2, er_slow=30)
    kama_1d_4h = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Get 12h data for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter (4h timeframe)
    vol_ma_15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (2.0 * vol_ma_15)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama_1d_4h[i]) or np.isnan(ema_50_12h_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above both KAMA and EMA50 with volume spike
            if close[i] > kama_1d_4h[i] and close[i] > ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below both KAMA and EMA50 with volume spike
            elif close[i] < kama_1d_4h[i] and close[i] < ema_50_12h_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals