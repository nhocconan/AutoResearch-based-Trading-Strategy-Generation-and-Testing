#!/usr/bin/env python3
# 4h_1d_1w_RSI_Divergence_With_Volume_Confirmation
# Hypothesis: RSI divergence (bullish/bearish) combined with volume confirmation and multi-timeframe trend (1d EMA50, 1w EMA200) 
# captures high-probability reversals in both bull and bear markets. Volume filters weak signals. 
# Target: 20-50 trades/year (80-200 total) to minimize fee drag.

name = "4h_1d_1w_RSI_Divergence_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for long-term trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 (trend filter)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1w EMA200 (long-term trend filter)
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate RSI on 4h data
    rsi = calculate_rsi(close, 14)
    
    # Calculate RSI divergence
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    lookback = 10
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence: lower low in price, higher low in RSI
        if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
            # Check if it's a meaningful divergence (price down, RSI up)
            price_change = (low[i] - low[i-lookback]) / low[i-lookback]
            rsi_change = rsi[i] - rsi[i-lookback]
            if price_change < -0.02 and rsi_change > 5:  # at least 2% price drop, 5 RSI points up
                bullish_div[i] = True
        
        # Bearish divergence: higher high in price, lower high in RSI
        if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
            price_change = (high[i] - high[i-lookback]) / high[i-lookback]
            rsi_change = rsi[i] - rsi[i-lookback]
            if price_change > 0.02 and rsi_change < -5:  # at least 2% price rise, 5 RSI points down
                bearish_div[i] = True
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish divergence + price above both EMAs + volume confirmation
            if (bullish_div[i] and 
                close[i] > ema50_1d_aligned[i] and 
                close[i] > ema200_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + price below both EMAs + volume confirmation
            elif (bearish_div[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  close[i] < ema200_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on bearish divergence or price below 1d EMA50
            if bearish_div[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on bullish divergence or price above 1d EMA50
            if bullish_div[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals