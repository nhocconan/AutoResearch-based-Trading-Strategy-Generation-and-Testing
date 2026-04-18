#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Use 1d KAMA to determine trend direction and RSI for momentum confirmation. Enter long when KAMA turns up and RSI > 50, short when KAMA turns down and RSI < 50. This strategy aims to capture medium-term trends while avoiding whipsaws in sideways markets. Designed for low trade frequency to minimize fee drag, with trend following that works in both bull and bear markets by adapting to price action.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle volatility calculation properly
        vol_sum = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - 9)
            vol_sum[i] = np.sum(np.abs(np.diff(close[start:i+1]))) if i > 0 else 0
        er = np.where(vol_sum > 0, change / vol_sum, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA
    kama_vals = kama(close)
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need KAMA and RSI warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama_vals[i]) or 
            np.isnan(rsi_vals[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: current vs previous
        kama_up = kama_vals[i] > kama_vals[i-1]
        kama_down = kama_vals[i] < kama_vals[i-1]
        rsi_val = rsi_vals[i]
        weekly_ema = ema_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, and above weekly EMA
            if kama_up and rsi_val > 50 and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, and below weekly EMA
            elif kama_down and rsi_val < 50 and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: KAMA turns down or RSI < 40
            if kama_down or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: KAMA turns up or RSI > 60
            if kama_up or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0