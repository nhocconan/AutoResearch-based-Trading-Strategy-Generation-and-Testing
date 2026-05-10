#!/usr/bin/env python3
# 6h_RSI_Divergence_1dTrend_Volume
# Hypothesis: RSI divergence on 6h with 1d trend filter and volume confirmation. 
# Long: Bullish RSI divergence + price above 1d EMA50 + volume > 1.5x average.
# Short: Bearish RSI divergence + price below 1d EMA50 + volume > 1.5x average.
# Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "6h_RSI_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    # Detect RSI divergence (lookback 10 periods)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    lookback = 10
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if it's a meaningful divergence (price lower, RSI higher)
            if low[i] == np.min(low[i-lookback:i+1]) and rsi[i] == np.max(rsi[i-lookback:i+1]):
                bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            # Check if it's a meaningful divergence (price higher, RSI lower)
            if high[i] == np.max(high[i-lookback:i+1]) and rsi[i] == np.min(rsi[i-lookback:i+1]):
                bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish RSI divergence + price above 1d EMA50 + volume confirmation
            if bullish_div[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish RSI divergence + price below 1d EMA50 + volume confirmation
            elif bearish_div[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish RSI divergence or price below EMA50
            if bearish_div[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish RSI divergence or price above EMA50
            if bullish_div[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals