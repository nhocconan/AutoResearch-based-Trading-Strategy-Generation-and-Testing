#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h RSI divergence filter and volume confirmation.
# Uses RSI divergence on 12h timeframe to detect weakening momentum before reversals,
# combined with 6h price action and volume to enter trades in the direction of the higher timeframe trend.
# Works in both bull and bear markets by filtering for exhaustion signals.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost.
name = "6h_12h_RSIDivergence_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for RSI calculation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate RSI on 12h timeframe
    rsi_12h = calculate_rsi(close_12h, 14)
    
    # Align RSI to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 6h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for EMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(rsi_12h_aligned[i]) or np.isnan(ema20[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 3:
            if (low[i] < low[i-2] and low[i-2] < low[i-4] and 
                rsi_12h_aligned[i] > rsi_12h_aligned[i-2] and rsi_12h_aligned[i-2] > rsi_12h_aligned[i-4]):
                bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 3:
            if (high[i] > high[i-2] and high[i-2] > high[i-4] and 
                rsi_12h_aligned[i] < rsi_12h_aligned[i-2] and rsi_12h_aligned[i-2] < rsi_12h_aligned[i-4]):
                bearish_div = True
        
        if position == 0:
            # Long when bullish divergence, price above EMA20, and volume confirmation
            if bullish_div and close[i] > ema20[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when bearish divergence, price below EMA20, and volume confirmation
            elif bearish_div and close[i] < ema20[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when bearish divergence or price falls below EMA20
            if bearish_div or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when bullish divergence or price rises above EMA20
            if bullish_div or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals