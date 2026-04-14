#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 10-period volume moving average (10*4h periods per day)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4-period high/low for breakout detection
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Get aligned indicators
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        vol_ma_10_val = vol_ma_10[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(rsi_1d_aligned) or np.isnan(vol_ma_10_val)):
            continue
        
        # Volume confirmation (> 1.2x average)
        volume_confirm = volume[i] > 1.2 * vol_ma_10_val
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: Break above 4-period high + RSI not overbought
                if high[i] > high_4[i] and rsi_1d_aligned < 70:
                    position = 1
                    signals[i] = position_size
                # Short: Break below 4-period low + RSI not oversold
                elif low[i] < low_4[i] and rsi_1d_aligned > 30:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when price breaks below 4-period low
            if low[i] < low_4[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price breaks above 4-period high
            if high[i] > high_4[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_4BarBreakout_RSI_Volume_v1"
timeframe = "4h"
leverage = 1.0