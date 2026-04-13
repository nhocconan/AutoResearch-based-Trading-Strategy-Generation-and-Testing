#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Bollinger Bands (20,2) + RSI(14) mean reversion
# Long when price touches lower Bollinger Band AND RSI < 30 (oversold)
# Short when price touches upper Bollinger Band AND RSI > 70 (overbought)
# Exit when price crosses middle Bollinger Band (20-period SMA)
# Uses daily timeframe for mean reversion structure to avoid whipsaws in trending markets
# Target: 75-200 total trades over 4 years (19-50/year) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20-period SMA, 2 standard deviations)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price at Bollinger Bands + RSI extreme
        at_lower_bb = close[i] <= lower_bb_aligned[i]
        at_upper_bb = close[i] >= upper_bb_aligned[i]
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        long_entry = at_lower_bb and rsi_oversold
        short_entry = at_upper_bb and rsi_overbought
        
        # Exit conditions: price crosses middle Bollinger Band
        exit_long = position == 1 and close[i] >= middle_bb_aligned[i]
        exit_short = position == -1 and close[i] <= middle_bb_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_bollinger_rsi_mean_reversion"
timeframe = "4h"
leverage = 1.0