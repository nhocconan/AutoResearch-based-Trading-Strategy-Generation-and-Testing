#!/usr/bin/env python3
"""
6h_RSI_Divergence_1dTrend_Filter
Hypothesis: Uses RSI divergence (bullish/bearish) on 6h chart for entry, filtered by 1d trend (price above/below EMA50). Exits on RSI opposite extreme or trend reversal. Designed to capture reversals in both bull and bear markets by combining momentum exhaustion with trend context. Targets 20-40 trades/year via strict divergence conditions and trend filter.
"""

name = "6h_RSI_Divergence_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_divergence(close, rsi, lookback=14):
    """Find bullish and bearish divergence"""
    bullish_div = np.zeros(len(close), dtype=bool)
    bearish_div = np.zeros(len(close), dtype=bool)
    
    for i in range(lookback, len(close)):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if close[i] < close[i-lookback:i].min() and rsi[i] > rsi[i-lookback:i].min():
            bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if close[i] > close[i-lookback:i].max() and rsi[i] < rsi[i-lookback:i].max():
            bearish_div[i] = True
            
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate RSI on 6h
    rsi_6h = calculate_rsi(close, 14)
    
    # Find divergences
    bullish_div, bearish_div = find_divergence(close, rsi_6h, 14)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi_6h[i]) or np.isnan(ema_50_6h[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish divergence + price above 1d EMA50
            if bullish_div[i] and close[i] > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + price below 1d EMA50
            elif bearish_div[i] and close[i] < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI > 70 (overbought) or trend turns down
                if rsi_6h[i] > 70 or close[i] < ema_50_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI < 30 (oversold) or trend turns up
                if rsi_6h[i] < 30 or close[i] > ema_50_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals