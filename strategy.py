#!/usr/bin/env python3
# 6h_RSI_Stochastic_Divergence_1dTrend
# Hypothesis: Combines RSI(14) and Stochastic(14,3,3) divergence on 6h with 1d EMA50 trend filter.
# Divergences signal exhaustion in current trend, allowing counter-trend entries with trend filter to avoid chop.
# Works in bull/bear: Trend filter ensures trades align with higher timeframe direction, reducing false signals.
# Uses RSI and Stochastic for early reversal detection, with divergence confirmation for higher accuracy.

name = "6h_RSI_Stochastic_Divergence_1dTrend"
timeframe = "6h"
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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic(14,3,3)
    lowest_low = np.full_like(low, np.nan)
    highest_high = np.full_like(high, np.nan)
    
    for i in range(len(low)):
        if i >= 13:
            lowest_low[i] = np.min(low[i-13:i+1])
            highest_high[i] = np.max(high[i-13:i+1])
    
    stoch_k = np.divide((close - lowest_low), (highest_high - lowest_low), 
                        out=np.full_like(close, np.nan), where=(highest_high - lowest_low)!=0) * 100
    
    stoch_k_smooth = np.full_like(stoch_k, np.nan)
    if len(stoch_k) >= 3:
        stoch_k_smooth[2] = np.mean(stoch_k[0:3])
        for i in range(3, len(stoch_k)):
            stoch_k_smooth[i] = (stoch_k_smooth[i-1] * 2 + stoch_k[i]) / 3
    
    stoch_d = np.full_like(stoch_k_smooth, np.nan)
    if len(stoch_k_smooth) >= 3:
        stoch_d[2] = np.mean(stoch_k_smooth[0:3])
        for i in range(3, len(stoch_k_smooth)):
            stoch_d[i] = (stoch_d[i-1] * 2 + stoch_k_smooth[i]) / 3
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Detect RSI and Stochastic divergences
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    lookback = 5
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (low[i] < low[i-lookback] and 
            rsi[i] > rsi[i-lookback] and 
            not np.isnan(rsi[i]) and not np.isnan(rsi[i-lookback])):
            bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (high[i] > high[i-lookback] and 
            rsi[i] < rsi[i-lookback] and 
            not np.isnan(rsi[i]) and not np.isnan(rsi[i-lookback])):
            bearish_div[i] = True
    
    # Also check Stochastic divergence
    for i in range(lookback, n):
        # Bullish stochastic divergence
        if (low[i] < low[i-lookback] and 
            stoch_d[i] > stoch_d[i-lookback] and 
            not np.isnan(stoch_d[i]) and not np.isnan(stoch_d[i-lookback])):
            bullish_div[i] = True
        # Bearish stochastic divergence
        if (high[i] > high[i-lookback] and 
            stoch_d[i] < stoch_d[i-lookback] and 
            not np.isnan(stoch_d[i]) and not np.isnan(stoch_d[i-lookback])):
            bearish_div[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, lookback)  # RSI(14) + Stochastic smoothing + lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(stoch_d[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish divergence AND uptrend (price > EMA50)
            if bullish_div[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish divergence AND downtrend (price < EMA50)
            elif bearish_div[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish divergence OR trend reversal (price < EMA50)
            if bearish_div[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish divergence OR trend reversal (price > EMA50)
            if bullish_div[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals