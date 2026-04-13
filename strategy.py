#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day KAMA trend filter + 1-day RSI mean reversion
# Long when KAMA is rising and RSI < 40 (oversold in uptrend)
# Short when KAMA is falling and RSI > 60 (overbought in downtrend)
# Exit when RSI crosses 50 (mean reversion) or KAMA changes direction
# Uses daily timeframe for trend structure to avoid whipsaws in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction (using previous bar to avoid look-ahead)
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # Entry conditions: KAMA direction + RSI extreme
        long_entry = kama_rising and rsi_aligned[i] < 40
        short_entry = kama_falling and rsi_aligned[i] > 60
        
        # Exit conditions: RSI crosses 50 or KAMA changes direction
        exit_long = position == 1 and (rsi_aligned[i] >= 50 or not kama_rising)
        exit_short = position == -1 and (rsi_aligned[i] <= 50 or not kama_falling)
        
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

name = "12h_1d_kama_rsi_mean_reversion"
timeframe = "12h"
leverage = 1.0