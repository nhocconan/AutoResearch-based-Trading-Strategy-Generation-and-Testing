#!/usr/bin/env python3
"""
4h_RSI_Divergence_1dTrend_Volume
Hypothesis: RSI divergence on 4h with daily trend filter and volume confirmation captures reversals in both bull and bear markets. RSI divergence signals exhaustion of momentum, and when combined with higher-timeframe trend and volume, provides high-probability reversals with low trade frequency.
"""
name = "4h_RSI_Divergence_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) on 4h
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            bull_div = False
            if i >= 3:
                # Look for divergence over last 3 bars
                if low[i] < low[i-1] and low[i-1] < low[i-2] and \
                   rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]:
                    bull_div = True
            
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            bear_div = False
            if i >= 3:
                if high[i] > high[i-1] and high[i-1] > high[i-2] and \
                   rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]:
                    bear_div = True
            
            # Long: bullish divergence + daily uptrend + volume
            if bull_div and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + daily downtrend + volume
            elif bear_div and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: opposite RSI divergence or trend change
            if position == 1:
                # Exit long on bearish divergence or price below EMA
                if bear_div or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short on bullish divergence or price above EMA
                if bull_div or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals