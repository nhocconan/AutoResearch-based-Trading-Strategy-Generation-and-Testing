#!/usr/bin/env python3
"""
4h_RSI_Divergence_Pullback
Hypothesis: Buy pullbacks in uptrend when RSI shows bullish divergence (higher low in RSI while price makes lower low) and vice versa for shorts. 
Uses 4h timeframe with 1d trend filter (EMA50) and volume confirmation. 
Designed for 20-30 trades/year to minimize fee drag. Works in bull/bear via trend filter.
"""

name = "4h_RSI_Divergence_Pullback"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4-period RSI for divergence detection
    delta4 = np.diff(close, prepend=close[0])
    gain4 = np.where(delta4 > 0, delta4, 0)
    loss4 = np.where(delta4 < 0, -delta4, 0)
    avg_gain4 = pd.Series(gain4).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss4 = pd.Series(loss4).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs4 = np.divide(avg_gain4, avg_loss4, out=np.zeros_like(avg_gain4), where=avg_loss4!=0)
    rsi4 = 100 - (100 / (1 + rs4))
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(rsi4[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for bullish divergence: price makes lower low, RSI makes higher low
            # Check last 5 bars for swing low
            lookback = 5
            if i >= lookback:
                # Find lowest low in lookback period
                lowest_low_idx = np.argmin(low[i-lookback:i+1]) + i - lookback
                # Find second lowest low (excluding the lowest)
                mask = np.ones(lookback+1, dtype=bool)
                mask[lowest_low_idx - (i - lookback)] = False
                if np.any(mask):
                    second_lowest_low_idx = np.argmin(low[i-lookback:i+1] * mask) + i - lookback
                    # Bullish divergence: price lower low but RSI higher low
                    if (low[lowest_low_idx] < low[second_lowest_low_idx] and 
                        rsi4[lowest_low_idx] > rsi4[second_lowest_low_idx] and
                        trend_up and vol_ratio[i] > 1.5):
                        signals[i] = 0.25
                        position = 1
                # Look for bearish divergence: price makes higher high, RSI makes lower high
                highest_high_idx = np.argmax(high[i-lookback:i+1]) + i - lookback
                mask = np.ones(lookback+1, dtype=bool)
                mask[highest_high_idx - (i - lookback)] = False
                if np.any(mask):
                    second_highest_high_idx = np.argmax(high[i-lookback:i+1] * mask) + i - lookback
                    # Bearish divergence: price higher high but RSI lower high
                    if (high[highest_high_idx] > high[second_highest_high_idx] and 
                        rsi4[highest_high_idx] < rsi4[second_highest_high_idx] and
                        trend_down and vol_ratio[i] > 1.5):
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: RSI overbought or trend turns down
            if rsi[i] > 70 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or trend turns up
            if rsi[i] < 30 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals