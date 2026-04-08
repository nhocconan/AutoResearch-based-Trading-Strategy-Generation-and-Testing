#!/usr/bin/env python3
# 12h_rsi_divergence_weekly_trend_v1
# Hypothesis: Use weekly trend filter (EMA200) with RSI(14) divergence on 12h timeframe to capture reversals in both bull and bear markets.
# Long when weekly EMA200 uptrend and RSI makes higher low while price makes lower low (bullish divergence).
# Short when weekly EMA200 downtrend and RSI makes lower high while price makes higher high (bearish divergence).
# Exit on opposite divergence signal or when price crosses weekly EMA50.
# Uses divergence to catch reversals with low frequency to minimize fee drag.
# Target: 15-25 trades/year to stay within limits while capturing major reversals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_rsi_divergence_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMAs to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI(14) on 12h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish divergence or price crosses below weekly EMA50
            bearish_div = False
            if i >= 3:
                # Check for bearish divergence: price higher high, RSI lower high
                if high[i] > high[i-1] and high[i-1] > high[i-2] and rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]:
                    bearish_div = True
            if bearish_div or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish divergence or price crosses above weekly EMA50
            bullish_div = False
            if i >= 3:
                # Check for bullish divergence: price lower low, RSI higher low
                if low[i] < low[i-1] and low[i-1] < low[i-2] and rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]:
                    bullish_div = True
            if bullish_div or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if i >= 3:
                if low[i] < low[i-1] and low[i-1] < low[i-2] and rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]:
                    bullish_div = True
            
            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if i >= 3:
                if high[i] > high[i-1] and high[i-1] > high[i-2] and rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]:
                    bearish_div = True
            
            # Weekly trend filter
            uptrend = close[i] > ema200_1w_aligned[i]
            downtrend = close[i] < ema200_1w_aligned[i]
            
            # Long entry: weekly uptrend and bullish divergence
            if bullish_div and uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: weekly downtrend and bearish divergence
            elif bearish_div and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals