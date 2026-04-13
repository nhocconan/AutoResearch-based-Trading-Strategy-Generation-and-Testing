#!/usr/bin/env python3
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4-period RSI on 1d (fast momentum)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4 = 100 - (100 / (1 + rs))
    
    # Calculate 20-period SMA on 1d (trend filter)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 10-period SMA on 1w (slow trend)
    sma_10_1w = pd.Series(close_1w).rolling(window=10, min_periods=10).mean().values
    
    # Align indicators to 4h timeframe
    rsi_4_aligned = align_htf_to_ltf(prices, df_1d, rsi_4)
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_4_aligned[i]) or 
            np.isnan(sma_20_1d_aligned[i]) or
            np.isnan(sma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions: extreme levels for mean reversion
        rsi_oversold = rsi_4_aligned[i] < 25
        rsi_overbought = rsi_4_aligned[i] > 75
        
        # Trend filters: price relative to SMAs
        above_sma20 = close[i] > sma_20_1d_aligned[i]
        below_sma20 = close[i] < sma_20_1d_aligned[i]
        above_sma10w = close[i] > sma_10_1w_aligned[i]
        below_sma10w = close[i] < sma_10_1w_aligned[i]
        
        # Entry conditions: mean reversion with trend alignment
        long_entry = rsi_oversold and above_sma20 and above_sma10w
        short_entry = rsi_overbought and below_sma20 and below_sma10w
        
        # Exit conditions: RSI returns to neutral or trend breaks
        exit_long = position == 1 and (rsi_4_aligned[i] > 60 or below_sma20)
        exit_short = position == -1 and (rsi_4_aligned[i] < 40 or above_sma20)
        
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

name = "4h_rsi4_sma20_1d_sma10_1w_mean_reversion"
timeframe = "4h"
leverage = 1.0