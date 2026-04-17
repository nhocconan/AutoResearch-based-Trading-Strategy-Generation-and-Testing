#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Trend
Strategy: Daily KAMA direction with RSI filter and weekly trend alignment.
Long: KAMA trending up + RSI < 40 + price > weekly EMA34
Short: KAMA trending down + RSI > 60 + price < weekly EMA34
Exit: KAMA changes direction or RSI crosses 50
Position size: 0.25
Designed to capture trend-following moves in both bull and bear markets with mean-reversion entries.
Timeframe: 1d
"""

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close_prices, fast=2, slow=30):
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, fast=2, slow=30)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_series_1w = pd.Series(close_1w)
    ema34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: current vs previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_exit = abs(rsi[i] - 50) < 5  # Exit when RSI near neutral
        
        if position == 0:
            # Long: KAMA up + RSI oversold + price above weekly EMA
            if kama_up and rsi_oversold and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + price below weekly EMA
            elif kama_down and rsi_overbought and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA down or RSI near neutral
            if kama_down or rsi_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up or RSI near neutral
            if kama_up or rsi_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Trend"
timeframe = "1d"
leverage = 1.0