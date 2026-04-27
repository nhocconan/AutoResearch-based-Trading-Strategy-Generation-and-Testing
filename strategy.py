#!/usr/bin/env python3
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
    
    # Get 1d data for 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate KAMA on 12h data
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 1-period volatility sum
    # Handle array alignment for volatility sum
    vol_sum = np.full_like(close_12h, np.nan)
    for i in range(1, len(close_12h)):
        vol_sum[i] = np.sum(np.abs(np.diff(close_12h[max(0, i-9):i+1], n=1)))
    er = np.where(vol_sum != 0, change / vol_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Get 1d data for RSI calculation
    rsi_period = 14
    if len(df_1d) < rsi_period + 1:
        return np.zeros(n)
    
    # Calculate RSI on 1d data
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    for i in range(rsi_period + 1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 12h data for Choppiness Index calculation
    chop_period = 14
    if len(df_12h) < chop_period * 2:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    atr = np.full_like(close_12h, np.nan)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    tr = np.insert(tr, 0, np.nan)
    for i in range(chop_period, len(atr)):
        atr[i] = np.nansum(tr[i-chop_period+1:i+1]) / chop_period
    
    max_high = np.full_like(close_12h, np.nan)
    min_low = np.full_like(close_12h, np.nan)
    for i in range(chop_period-1, len(close_12h)):
        max_high[i] = np.max(high_12h[i-chop_period+1:i+1])
        min_low[i] = np.min(low_12h[i-chop_period+1:i+1])
    
    chop = np.full_like(close_12h, np.nan)
    for i in range(chop_period-1, len(close_12h)):
        if max_high[i] != min_low[i] and atr[i] > 0:
            chop[i] = 100 * np.log10(sum(atr[i-chop_period+1:i+1]) / (max_high[i] - min_low[i])) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(50, 34, rsi_period + 1, chop_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(kama_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # KAMA direction
        kama_up = price > kama_aligned[i]
        kama_down = price < kama_aligned[i]
        
        # RSI conditions
        rsi_overbought = rsi_aligned[i] > 70
        rsi_oversold = rsi_aligned[i] < 30
        
        # Choppy market filter (avoid trending markets)
        choppy = chop_aligned[i] > 50  # Chop > 50 indicates ranging market
        
        if position == 0:
            # Long entry: KAMA up + RSI oversold + choppy market (mean reversion in range)
            if kama_up and rsi_oversold and choppy:
                signals[i] = size
                position = 1
            # Short entry: KAMA down + RSI overbought + choppy market (mean reversion in range)
            elif kama_down and rsi_overbought and choppy:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: KAMA down or RSI overbought or trend change
            if kama_down or rsi_overbought or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: KAMA up or RSI oversold or trend change
            if kama_up or rsi_oversold or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_Chop_MeanReversion"
timeframe = "12h"
leverage = 1.0