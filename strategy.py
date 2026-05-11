#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly KAMA for trend direction
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i-1] * (close_1w[i] - kama[i-1])
    # Trend filter
    trend_up_1w = close_1w > kama
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Daily KAMA for entry signal
    change_d = np.abs(np.diff(close, k=2))
    volatility_d = np.sum(np.abs(np.diff(close)), axis=0)
    er_d = np.divide(change_d, volatility_d, out=np.zeros_like(change_d), where=volatility_d!=0)
    sc_d = (er_d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_d = np.zeros_like(close)
    kama_d[0] = close[0]
    for i in range(1, len(close)):
        kama_d[i] = kama_d[i-1] + sc_d[i-1] * (close[i] - kama_d[i-1])
    
    # Daily RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chopiness index (14-period)
    atr = pd.Series(np.maximum(high - low, np.maximum(high - np.roll(close, 1), np.roll(close, 1) - low))).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_d[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + chop < 61.8 (trending) + weekly trend up + volume
            if close[i] > kama_d[i] and rsi[i] > 50 and chop[i] < 61.8 and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + chop < 61.8 + weekly trend down + volume
            elif close[i] < kama_d[i] and rsi[i] < 50 and chop[i] < 61.8 and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down OR chop > 61.8 (choppy)
            if close[i] < kama_d[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up OR chop > 61.8
            if close[i] > kama_d[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals