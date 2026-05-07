#!/usr/bin/env python3
# 1h_RSI_OverboughtOversold_4hTrend_1dVolume
# Hypothesis: 1-hour RSI mean reversion with 4-hour trend filter and 1-day volume confirmation.
# Works in bull/bear markets by using RSI extremes only when aligned with higher timeframe trend.
# Long: RSI < 30 on 1h, price above 4h EMA50 (uptrend), volume > 1.5x 20-day average.
# Short: RSI > 70 on 1h, price below 4h EMA50 (downtrend), volume > 1.5x 20-day average.
# Exit: RSI returns to neutral zone (40-60) or opposing extreme triggers reversal.
# Designed for 15-30 trades/year with controlled risk via trend alignment and volume filter.

name = "1h_RSI_OverboughtOversold_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1-day average volume (20-day for stability)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1-hour RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 50, 20)  # Ensure we have RSI, EMA50, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30), price above 4h EMA50 (uptrend), volume spike
            if (rsi[i] < 30 and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70), price below 4h EMA50 (downtrend), volume spike
            elif (rsi[i] > 70 and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral (>=40) or reverses to overbought
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: RSI returns to neutral (<=60) or reverses to oversold
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals