#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1w EMA trend filter, 1d KAMA trend, and volume confirmation.
# Long when 1w trend up, price above 1d KAMA, and volume > 1.3x average.
# Short when 1w trend down, price below 1d KAMA, and volume > 1.3x average.
# Uses discrete position sizing (0.25) to minimize churn and manage drawdown.
# Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "12h_1wEMA_1dKAMA_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1w EMA(34) for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = ema_34_1w > np.roll(ema_34_1w, 1)
    trend_1w_up = np.where(np.isnan(trend_1w_up), False, trend_1w_up)
    
    # 1d KAMA(30, 2, 30)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    # Align 1d KAMA to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume average (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w trend up, price above 1d KAMA, volume spike
            if (trend_1w_up_aligned[i] and
                close[i] > kama_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: 1w trend down, price below 1d KAMA, volume spike
            elif (not trend_1w_up_aligned[i] and
                  close[i] < kama_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend break or price below KAMA
            if (not trend_1w_up_aligned[i] or close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend break or price above KAMA
            if (trend_1w_up_aligned[i] or close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals