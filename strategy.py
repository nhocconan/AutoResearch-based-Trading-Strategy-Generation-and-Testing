#!/usr/bin/env python3
name = "1d_WeeklyKAMA_Trend_12hRSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly KAMA Trend (Higher Timeframe) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_1w, n=10))
    volatility = np.nansum(np.abs(np.diff(close_1w, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]  # Seed
    for i in range(10, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc[i-1] * (close_1w[i] - kama_1w[i-1])
    
    # Align KAMA to daily
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === 12h RSI Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # RSI calculation
    delta = np.diff(close_12h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = np.concatenate([np.full(14, np.nan), rsi_12h])
    
    # Align RSI to daily
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # === Daily Price for Entry/Exit ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly KAMA (bullish trend) and 12h RSI not overbought
            if (close[i] > kama_1w_aligned[i] and 
                rsi_12h_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly KAMA (bearish trend) and 12h RSI not oversold
            elif (close[i] < kama_1w_aligned[i] and 
                  rsi_12h_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly KAMA or RSI overbought
            if close[i] < kama_1w_aligned[i] or rsi_12h_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly KAMA or RSI oversold
            if close[i] > kama_1w_aligned[i] or rsi_12h_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals