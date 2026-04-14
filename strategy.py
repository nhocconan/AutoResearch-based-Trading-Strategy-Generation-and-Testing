#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Williams %R and 1d RSI filter
# Williams %R identifies overbought/oversold conditions on 12h timeframe
# RSI on 1d provides trend filter: only take Williams %R signals when 1d RSI > 50 (bullish bias) or < 50 (bearish bias)
# Works in both bull and bear markets as it adapts to the higher timeframe trend
# Low frequency: Williams %R signals are infrequent, reducing overtrading

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 12h data ONCE for Williams %R
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Williams %R (14 periods)
    willr_length = 14
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Highest high and lowest low over willr_length
    highest_high = pd.Series(high_12h).rolling(window=willr_length, min_periods=willr_length).max().values
    lowest_low = pd.Series(low_12h).rolling(window=willr_length, min_periods=willr_length).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    willr = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)  # Handle division by zero
    
    # Align Williams %R to 6h timeframe
    willr_aligned = align_htf_to_ltf(prices, df_12h, willr)
    
    # Load 1d data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI (14 periods)
    rsi_length = 14
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # When no losses, RSI = 100
    rsi = np.where(avg_gain == 0, 0, rsi)    # When no gains, RSI = 0
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 14)  # Need enough for Williams %R and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(willr_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        willr_val = willr_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) AND 1d RSI > 50 (bullish bias)
            if willr_val < -80 and rsi_val > 50:
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought (> -20) AND 1d RSI < 50 (bearish bias)
            elif willr_val > -20 and rsi_val < 50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R becomes overbought (> -20) OR 1d RSI < 40 (loss of bullish momentum)
            if willr_val > -20 or rsi_val < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R becomes oversold (< -80) OR 1d RSI > 60 (loss of bearish momentum)
            if willr_val < -80 or rsi_val > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hWilliamsR_1dRSI_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0