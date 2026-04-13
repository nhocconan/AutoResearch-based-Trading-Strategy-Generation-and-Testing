#!/usr/bin/env python3
"""
4h_1d_RSI_Divergence_Pullback
Hypothesis: Buy pullbacks in uptrend when RSI shows bullish divergence from oversold levels,
sell rallies in downtrend when RSI shows bearish divergence from overbought levels.
Uses 1d trend filter (price above/below 200 EMA) to ensure trades align with higher timeframe direction.
RSI divergence is calculated on 4h chart but only acts when 1d trend confirms.
Target: 20-30 trades/year with controlled risk via RSI-based exit.
Works in bull markets by buying dips, in bear markets by selling rallies.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def detect_divergence(price, rsi, lookback=5):
    """
    Detect bullish/bearish RSI divergence.
    Returns: 1 for bullish divergence, -1 for bearish divergence, 0 otherwise
    """
    n = len(price)
    divergence = np.zeros(n)
    
    for i in range(lookback, n):
        # Look for bullish divergence: price makes lower low, RSI makes higher low
        if i >= lookback:
            price_low = np.min(price[i-lookback:i+1])
            price_prev_low = np.min(price[i-2*lookback:i-lookback+1]) if i >= 2*lookback else price_low
            rsi_low = np.min(rsi[i-lookback:i+1])
            rsi_prev_low = np.min(rsi[i-2*lookback:i-lookback+1]) if i >= 2*lookback else rsi_low
            
            bullish_div = (price_low < price_prev_low) and (rsi_low > rsi_prev_low)
            
            # Look for bearish divergence: price makes higher high, RSI makes lower high
            price_high = np.max(price[i-lookback:i+1])
            price_prev_high = np.max(price[i-2*lookback:i-lookback+1]) if i >= 2*lookback else price_high
            rsi_high = np.max(rsi[i-lookback:i+1])
            rsi_prev_high = np.max(rsi[i-2*lookback:i-lookback+1]) if i >= 2*lookback else rsi_high
            
            bearish_div = (price_high > price_prev_high) and (rsi_high < rsi_prev_high)
            
            if bullish_div:
                divergence[i] = 1
            elif bearish_div:
                divergence[i] = -1
    
    return divergence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter (200 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d 200 EMA for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h RSI for divergence detection
    rsi_4h = calculate_rsi(close, 14)
    
    # Detect RSI divergence on 4h chart
    divergence = detect_divergence(close, rsi_4h, 5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if trend filter not ready
        if np.isnan(ema_200_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend: price above/below 200 EMA
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Long: bullish divergence in uptrend (buy pullbacks)
        long_condition = (divergence[i] == 1) and uptrend
        
        # Short: bearish divergence in downtrend (sell rallies)
        short_condition = (divergence[i] == -1) and downtrend
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_RSI_Divergence_Pullback"
timeframe = "4h"
leverage = 1.0