#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_RSI_1dChop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index on 1d (14-period)
    atr_1d = np.zeros_like(close_1d)
    tr1 = np.abs(np.diff(high_1d := df_1d['high'].values, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d := df_1d['low'].values, prepend=low_1d[0]))
    tr3 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3[0] = tr2[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    # Align to 4h
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_4h = align_htf_to_ltf(prices, df_1d, rsi)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = kama_4h[i]
        rsi_val = rsi_4h[i]
        chop_val = chop_4h[i]
        
        # Chop regime: > 61.8 = range, < 38.2 = trend
        in_range = chop_val > 61.8
        in_trend = chop_val < 38.2
        
        if position == 0:
            # Long: KAMA uptrend + RSI not overbought + in trending regime
            if close[i] > trend and rsi_val < 70 and in_trend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA downtrend + RSI not oversold + in trending regime
            elif close[i] < trend and rsi_val > 30 and in_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend reversal or RSI overbought
            if close[i] < trend or rsi_val > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reversal or RSI oversold
            if close[i] > trend or rsi_val < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals