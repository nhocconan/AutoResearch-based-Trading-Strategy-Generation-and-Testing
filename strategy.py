#!/usr/bin/env python3
"""
12h_RSI_Divergence_1dTrend
Hypothesis: RSI divergence signals on 12h timeframe filtered by 1d EMA50 trend work in both bull and bear markets.
Focus on quality over quantity: only trade when RSI shows clear divergence with price action.
Target: 15-25 trades per year to minimize fee drag while maintaining edge.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h RSI(14) for divergence detection
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and EMA
    start_idx = 55  # Enough for RSI(14) + EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        current_rsi = rsi_values[i]
        
        # Look back for divergence detection (check last 5 bars for pivot points)
        lookback = 5
        if i < lookback:
            signals[i] = size if position == 1 else (-size if position == -1 else 0.0)
            continue
            
        # Find recent swing highs and lows in price
        recent_high_idx = i - np.argmax(high[i-lookback:i+1][::-1]) if i >= lookback else i
        recent_low_idx = i - np.argmin(low[i-lookback:i+1][::-1]) if i >= lookback else i
        
        # Check for bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= lookback * 2:  # Need enough history
            # Find two recent swing highs
            highs = []
            for j in range(i - lookback, i + 1):
                if j >= 2 and j < n - 2:  # Avoid edges
                    if high[j] >= high[j-1] and high[j] >= high[j+1] and \
                       high[j] >= high[j-2] and high[j] >= high[j+2]:
                        highs.append(j)
            
            if len(highs) >= 2:
                # Get two most recent swing highs
                high1, high2 = highs[-2], highs[-1]
                # Bearish divergence: price higher high, RSI lower high
                if (high[high2] > high[high1] and 
                    rsi_values[high2] < rsi_values[high1]):
                    bearish_div = True
        
        # Check for bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= lookback * 2:  # Need enough history
            # Find two recent swing lows
            lows = []
            for j in range(i - lookback, i + 1):
                if j >= 2 and j < n - 2:  # Avoid edges
                    if low[j] <= low[j-1] and low[j] <= low[j+1] and \
                       low[j] <= low[j-2] and low[j] <= low[j+2]:
                        lows.append(j)
            
            if len(lows) >= 2:
                # Get two most recent swing lows
                low1, low2 = lows[-2], lows[-1]
                # Bullish divergence: price lower low, RSI higher low
                if (low[low2] < low[low1] and 
                    rsi_values[low2] > rsi_values[low1]):
                    bullish_div = True
        
        if position == 0:
            # Enter long on bullish divergence with uptrend
            if bullish_div and close[i] > ema_trend and current_rsi < 40:
                signals[i] = size
                position = 1
            # Enter short on bearish divergence with downtrend
            elif bearish_div and close[i] < ema_trend and current_rsi > 60:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bearish divergence or trend breaks down
            if bearish_div or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bullish divergence or trend breaks up
            if bullish_div or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI_Divergence_1dTrend"
timeframe = "12h"
leverage = 1.0