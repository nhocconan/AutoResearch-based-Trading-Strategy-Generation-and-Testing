#!/usr/bin/env python3
"""
1h_mean_reversion_v1
Hypothesis: Mean reversion on 1h timeframe works in both bull and bear markets by buying oversold conditions and selling overbought conditions. Uses 4h trend filter to avoid counter-trend trades and RSI(2) for extreme mean reversion signals. Volume confirmation filters out low-quality signals. Target: 15-30 trades/year to minimize fee drag while capturing mean reversion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(2) for extreme mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    
    # Volume filter: above average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(2, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(sma_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI < 10 (extremely oversold) + price below 4h SMA50
                if rsi[i] < 10 and close[i] < sma_50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: RSI > 90 (extremely overbought) + price above 4h SMA50
                elif rsi[i] > 90 and close[i] > sma_50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals