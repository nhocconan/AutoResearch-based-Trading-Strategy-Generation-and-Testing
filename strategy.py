#!/usr/bin/env python3
name = "6h_RSI_Divergence_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 14-period RSI on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 6-period RSI for faster signals
    avg_gain_fast = pd.Series(gain).ewm(alpha=1/6, adjust=False).mean().values
    avg_loss_fast = pd.Series(loss).ewm(alpha=1/6, adjust=False).mean().values
    rs_fast = avg_gain_fast / (avg_loss_fast + 1e-10)
    rsi_fast = 100 - (100 / (1 + rs_fast))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(rsi_fast[i]) or
            np.isnan(rsi[i-14]) or np.isnan(rsi_fast[i-6])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # RSI divergence detection
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        bullish_div = False
        bearish_div = False
        
        # Check for bullish divergence (need at least 14 periods lookback)
        if i >= 14:
            # Price lower low
            price_lower_low = close[i] < close[i-14] and low[i] < low[i-14]
            # RSI higher low
            rsi_higher_low = rsi[i] > rsi[i-14]
            bullish_div = price_lower_low and rsi_higher_low
        
        # Check for bearish divergence (need at least 6 periods lookback for fast RSI)
        if i >= 6:
            # Price higher high
            price_higher_high = close[i] > close[i-6] and high[i] > high[i-6]
            # RSI lower high
            rsi_lower_high = rsi_fast[i] < rsi_fast[i-6]
            bearish_div = price_higher_high and rsi_lower_high
        
        if position == 0:
            # Long: bullish divergence AND weekly uptrend
            if bullish_div and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence AND weekly downtrend
            elif bearish_div and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish divergence OR price below weekly EMA
            if bearish_div or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: bullish divergence OR price above weekly EMA
            if bullish_div or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals