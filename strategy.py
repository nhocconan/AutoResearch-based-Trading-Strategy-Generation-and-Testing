#!/usr/bin/env python3
# 4h_Weekly_Momentum_With_Volume_And_Trend_Filter
# Hypothesis: Uses 1-week momentum to capture major trend direction, entering only when
# 4h price breaks above/below weekly high/low with volume confirmation and 4h RSI filter.
# Weekly momentum provides strong trend filter that works in both bull and bear markets,
# while volume and RSI prevent entries in choppy conditions. Designed for low trade frequency.

name = "4h_Weekly_Momentum_With_Volume_And_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly momentum
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly high and low for breakout levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to 4h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # RSI(14) for 4h timeframe - avoid choppy markets
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high, RSI > 50 (bullish bias), volume confirmation
            if (close[i] > weekly_high_aligned[i] and 
                rsi[i] > 50 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low, RSI < 50 (bearish bias), volume confirmation
            elif (close[i] < weekly_low_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below weekly low (trend reversal)
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above weekly high (trend reversal)
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals