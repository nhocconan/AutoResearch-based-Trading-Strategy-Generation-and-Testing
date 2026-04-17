#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1d ATR (used for volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on daily timeframe
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR for position sizing scaling
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]  # First TR
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h SMA of close for trend filter
    sma_12h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for SMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_12h[i]) or np.isnan(sma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1d ATR is above its 50-period average (high volatility regime)
        atr_ma50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma50[i]):
            volatility_filter = False
        else:
            volatility_filter = atr_1d_aligned[i] > atr_ma50[i]
        
        # Trend filter: price above/below 20-period SMA
        trend_long = close[i] > sma_12h[i]
        trend_short = close[i] < sma_12h[i]
        
        if position == 0:
            # Long entry: price above SMA and high volatility regime
            if trend_long and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price below SMA and high volatility regime
            elif trend_short and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below SMA or volatility drops
            if not trend_long or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above SMA or volatility drops
            if not trend_short or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Volatility_Trend_Follow"
timeframe = "12h"
leverage = 1.0