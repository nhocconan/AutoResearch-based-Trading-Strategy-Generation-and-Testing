#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 1w
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w: Calculate KAMA for trend direction ===
    close_1w = df_1w['close'].values
    
    # Efficiency ratio for KAMA
    change = np.abs(np.diff(close_1w, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, 1)), axis=0)  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # === 1d: Calculate RSI(14) ===
    close_1d = prices['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d: Calculate Choppiness Index(14) ===
    high = prices['high'].values
    low = prices['low'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14)
    atr = np.full_like(close_1d, np.nan)
    atr[14] = np.nanmean(tr[1:15])
    for i in range(15, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR over 14 periods
    sum_atr = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        sum_atr[i] = np.sum(atr[i-13:i+1])
    
    # Max(high) - Min(low) over 14 periods
    max_high = np.full_like(close_1d, np.nan)
    min_low = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        max_high[i] = np.max(high[i-13:i+1])
        min_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if sum_atr[i] > 0 and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # Neutral when undefined
    
    # Align 1w KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = close_1d[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Above KAMA (uptrend), RSI oversold, low chop (trending market)
            if (close_val > kama_val and    # Above weekly KAMA (uptrend)
                rsi_val < 30 and            # Oversold RSI
                chop_val < 38.2):           # Trending market (low chop)
                signals[i] = 0.25
                position = 1
            # Short: Below KAMA (downtrend), RSI overbought, low chop (trending market)
            elif (close_val < kama_val and  # Below weekly KAMA (downtrend)
                  rsi_val > 70 and          # Overbought RSI
                  chop_val < 38.2):         # Trending market (low chop)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Below KAMA or RSI overbought or high chop (choppy market)
            if (close_val < kama_val or 
                rsi_val > 70 or 
                chop_val > 61.8):  # Choppy market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Above KAMA or RSI oversold or high chop (choppy market)
            if (close_val > kama_val or 
                rsi_val < 30 or 
                chop_val > 61.8):  # Choppy market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals