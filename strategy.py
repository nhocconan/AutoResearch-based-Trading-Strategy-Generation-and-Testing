#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d/1w structure for directional bias and 4h price action for entries.
# Uses 1d EMA200 for long-term trend and 1w ATR for volatility-based entry triggers.
# Designed for low trade frequency (<50/year) to minimize fee drag in 4h timeframe.
# Works in both bull/bear markets by combining trend filter with volatility breakout logic.
name = "4h_1dEMA200_1wATR_Breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Get 1w data for ATR calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1w ATR(14) for volatility-based entry
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_4h = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate 4h Donchian(20) channels for breakout signals
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Wait for EMA200 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(high_4h[i]) or np.isnan(low_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: require minimum ATR to avoid choppy markets
        vol_filter = atr_14_4h[i] > 0.01 * close[i]  # At least 1% of price
        
        if position == 0:
            # Long: price breaks above 4h Donchian high, above 1d EMA200, with volatility
            if (close[i] > high_4h[i] and 
                close[i] > ema_200_4h[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h Donchian low, below 1d EMA200, with volatility
            elif (close[i] < low_4h[i] and 
                  close[i] < ema_200_4h[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low or trend reversal
            if close[i] < low_4h[i] or close[i] < ema_200_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high or trend reversal
            if close[i] > high_4h[i] or close[i] > ema_200_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals