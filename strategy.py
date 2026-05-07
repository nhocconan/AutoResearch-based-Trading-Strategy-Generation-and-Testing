#!/usr/bin/env python3
"""
12h_RSI_Divergence_With_Trend_and_Volume
Hypothesis: Uses RSI divergence on 12h timeframe with 1d EMA50 trend filter and volume spike (>2x average) for confirmation. Designed for low trade frequency (12-37/year) to avoid fee drag. Works in both bull and bear markets by combining momentum divergence with trend alignment and volume confirmation. Uses discrete position sizing (0.25) to minimize churn.
"""

name = "12h_RSI_Divergence_With_Trend_and_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI on 12h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    # Find RSI divergence points
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    # Look for bullish divergence: price makes lower low, RSI makes higher low
    for i in range(20, n):
        # Find recent swing low in price
        if low[i] == np.min(low[i-19:i+1]):
            # Look back for previous swing low
            for j in range(i-20, max(0, i-40), -1):
                if low[j] == np.min(low[j-19:j+1]) and j < i-10:
                    if low[i] < low[j] and rsi[i] > rsi[j]:
                        bullish_div[i] = True
                    break
        
        # Find recent swing high in price
        if high[i] == np.max(high[i-19:i+1]):
            # Look back for previous swing high
            for j in range(i-20, max(0, i-40), -1):
                if high[j] == np.max(high[j-19:j+1]) and j < i-10:
                    if high[i] > high[j] and rsi[i] < rsi[j]:
                        bearish_div[i] = True
                    break
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: bullish RSI divergence, volume spike, price above EMA50
            if (bullish_div[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence, volume spike, price below EMA50
            elif (bearish_div[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish RSI divergence or trend changes
            if bearish_div[i] or not daily_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish RSI divergence or trend changes
            if bullish_div[i] or not daily_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals