#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1week trend filter using weekly moving average.
# Long: Price crosses above 10-day SMA + weekly close > weekly SMA(10) (bullish trend filter).
# Short: Price crosses below 10-day SMA + weekly close < weekly SMA(10) (bearish trend filter).
# Uses daily price action for entry with weekly trend filter to avoid counter-trend trades.
# Position size: 0.25 (25%) to manage drawdown in volatile markets.
# Target: 15-30 trades per year (60-120 over 4 years) for low frequency, high quality signals.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 10-day SMA for daily entry signal
    sma_10 = np.full(n, np.nan)
    for i in range(10, n):
        sma_10[i] = np.mean(close[i-10:i])
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    # Weekly SMA(10) for trend filter
    weekly_sma_10 = np.full(len(weekly_close), np.nan)
    for i in range(10, len(weekly_close)):
        weekly_sma_10[i] = np.mean(weekly_close[i-10:i])
    
    # Align weekly SMA to daily
    weekly_sma_10_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma_10)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(10, n):
        # Skip if any required data is not ready
        if (np.isnan(sma_10[i]) or np.isnan(weekly_sma_10_aligned[i]) or 
            np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma = sma_10[i]
        weekly_sma = weekly_sma_10_aligned[i]
        weekly_price = weekly_close_aligned[i]
        
        if position == 0:
            # Long: price crosses above daily SMA(10) + weekly close > weekly SMA(10)
            if (price > sma and weekly_price > weekly_sma):
                position = 1
                signals[i] = position_size
            # Short: price crosses below daily SMA(10) + weekly close < weekly SMA(10)
            elif (price < sma and weekly_price < weekly_sma):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below daily SMA(10)
            if price < sma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above daily SMA(10)
            if price > sma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_SMA10_Trend_Filter"
timeframe = "1d"
leverage = 1.0