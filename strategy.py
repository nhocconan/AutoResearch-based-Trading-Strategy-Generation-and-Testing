#!/usr/bin/env python3
# 4h_rsi_divergence_macd_12h_trend
# Hypothesis: RSI divergence combined with MACD crossover on 4h, filtered by 12h EMA trend.
# Long when bullish RSI divergence + MACD bullish crossover + price > 12h EMA50.
# Short when bearish RSI divergence + MACD bearish crossover + price < 12h EMA50.
# Exit when RSI crosses opposite extreme (50) or MACD reverses.
# Designed to capture momentum reversals with trend alignment in both bull and bear markets.
# Target: 80-150 total trades over 4 years (~20-38/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_divergence_macd_12h_trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate RSI (14-period) for divergence detection
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate MACD (12,26,9)
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    macd_line = macd_line.values
    signal_line = signal_line.values
    macd_hist = macd_hist.values
    
    # Detect RSI divergence (lookback 5 periods)
    rsi_divergence_bull = np.zeros(n, dtype=bool)
    rsi_divergence_bear = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-5]) or np.isnan(close[i]) or np.isnan(close[i-5]):
            continue
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if close[i] < close[i-5] and rsi[i] > rsi[i-5]:
            # Check if recent low is significant
            if low[i] <= np.min(low[i-4:i+1]) and low[i-5] <= np.min(low[i-9:i-4]):
                rsi_divergence_bull[i] = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if close[i] > close[i-5] and rsi[i] < rsi[i-5]:
            # Check if recent high is significant
            if high[i] >= np.max(high[i-4:i+1]) and high[i-5] >= np.max(high[i-9:i-4]):
                rsi_divergence_bear[i] = True
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(macd_line[i]) or np.isnan(signal_line[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or MACD bearish crossover
            if rsi[i] < 50 or (macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or MACD bullish crossover
            if rsi[i] > 50 or (macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry conditions
            macd_bullish = macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]
            macd_bearish = macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]
            
            # Long: bullish RSI divergence + MACD bullish crossover + uptrend
            if (rsi_divergence_bull[i] and macd_bullish and 
                close[i] > ema_50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish RSI divergence + MACD bearish crossover + downtrend
            elif (rsi_divergence_bear[i] and macd_bearish and 
                  close[i] < ema_50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals