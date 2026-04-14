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
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 6-period average volume (4h periods in a day for 6h TF)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned indicators
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        vol_ma_4_val = vol_ma_4[i]  # already LTF
        
        # Check for NaN values
        if np.isnan(rsi_1d_aligned) or np.isnan(vol_ma_4_val):
            continue
        
        # Volume confirmation (> 1.2x average)
        volume_confirm = volume[i] > 1.2 * vol_ma_4_val
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: Oversold (RSI < 30)
                if rsi_1d_aligned < 30:
                    position = 1
                    signals[i] = position_size
                # Short: Overbought (RSI > 70)
                elif rsi_1d_aligned > 70:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI overbought
            if rsi_1d_aligned > 70:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI oversold
            if rsi_1d_aligned < 30:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_RSI30_70_Volume1.2x_v1"
timeframe = "6h"
leverage = 1.0