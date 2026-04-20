#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) - 10 period
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = np.nan
    kama_up = kama > kama_prev
    kama_down = kama < kama_prev
    
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_up)
    kama_down_aligned = align_htf_to_ltf(prices, df_1d, kama_down)
    
    # Calculate 1d RSI (14) for mean reversion
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d chopiness index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    atr1 = np.abs(high_1d - low_1d)
    atr2 = np.abs(high_1d - np.roll(close_1d, 1))
    atr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    chop = np.where(range_14 != 0, 100 * np.log10(atr_sum / range_14) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        kama_up_val = kama_up_aligned[i]
        kama_down_val = kama_down_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_up_val) or np.isnan(kama_down_val) or 
            np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI oversold + chop not extreme
            if kama_up_val and rsi_val < 35 and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + chop not extreme
            elif kama_down_val and rsi_val > 65 and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down or RSI overbought
            if kama_down_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up or RSI oversold
            if kama_up_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_KAMA_RSI_ChopFilter_V1
# Uses 1-day KAMA for trend direction (adaptive moving average)
# Enters long when KAMA turns up + RSI < 35 (oversold) + chop < 61.8 (not too trendy)
# Enters short when KAMA turns down + RSI > 65 (overbought) + chop < 61.8
# Exits when KAMA reverses or RSI reaches extreme levels
# Chop filter avoids whipsaws in strong trends
# Designed for 4h timeframe with ~20-50 trades/year
name = "4h_KAMA_RSI_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0