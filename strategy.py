#!/usr/bin/env python3
# [24990] 1d_1w_kama_rsi_chop_v1
# Hypothesis: 1-day KAMA direction + RSI filter + Choppiness regime filter. Long when KAMA rising, RSI < 60, and chop > 61.8 (ranging). Short when KAMA falling, RSI > 40, and chop > 61.8. Exit when opposite signal. Designed to capture mean reversion in ranging markets while avoiding trends. Uses weekly trend filter to avoid counter-trend trades in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA (10, 2, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    
    for i in range(15, n):
        if np.isnan(avg_gain[i-1]):
            avg_gain[i] = np.nanmean(gain[i-13:i+1])
            avg_loss[i] = np.nanmean(loss[i-13:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    atr = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr[i] = np.mean(tr[max(0, i-13):i+1])
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / (atr[i] * (highest_high[i] - lowest_low[i]))) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: KAMA falling OR RSI > 60 OR chop < 61.8 (trending)
            if kama[i] < kama[i-1] or rsi[i] > 60 or chop[i] < 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: KAMA rising OR RSI < 40 OR chop < 61.8 (trending)
            if kama[i] > kama[i-1] or rsi[i] < 40 or chop[i] < 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA rising, RSI < 60, chop > 61.8 (ranging), and price above weekly EMA50 (avoid strong downtrend)
            if kama[i] > kama[i-1] and rsi[i] < 60 and chop[i] > 61.8 and close[i] > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA falling, RSI > 40, chop > 61.8 (ranging), and price below weekly EMA50 (avoid strong uptrend)
            elif kama[i] < kama[i-1] and rsi[i] > 40 and chop[i] > 61.8 and close[i] < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals