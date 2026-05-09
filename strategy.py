#!/usr/bin/env python3
# 1h_4h1d_Trend_Filtered_Momentum
# Strategy: 1h momentum with 4h trend filter and 1d volatility filter
# Long when 1h RSI > 55, 4h EMA20 uptrend, and 1d ATR ratio < 1.2 (low volatility)
# Short when 1h RSI < 45, 4h EMA20 downtrend, and 1d ATR ratio < 1.2
# Exit when RSI crosses back to 50 or volatility spikes
# Designed for 1h timeframe with strict filters to limit trades to 15-35/year
# Works in bull/bear by following 4h trend and avoiding high-volatility chop

name = "1h_4h1d_Trend_Filtered_Momentum"
timeframe = "1h"
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
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])))
    tr2 = np.maximum(np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]])), tr1)
    atr_1d = pd.Series(tr2).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ratio = atr_1d / atr_ma_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_smooth = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_smooth = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI bullish, 4h uptrend, low volatility
            if rsi[i] > 55 and ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and atr_ratio_aligned[i] < 1.2:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI bearish, 4h downtrend, low volatility
            elif rsi[i] < 45 and ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and atr_ratio_aligned[i] < 1.2:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses below 50 or volatility spikes
            if rsi[i] < 50 or atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI crosses above 50 or volatility spikes
            if rsi[i] > 50 or atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals