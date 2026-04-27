#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Weekly ATR for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily ATR(14) for volatility measurement
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Weekly ATR(14) for volatility filter
    high_low_w = df_1w['high'] - df_1w['low']
    high_close_w = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    low_close_w = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr_w = np.maximum(high_low_w, np.maximum(high_close_w, low_close_w))
    atr_14_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_14_w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_w)
    
    # Daily 200-period SMA for long-term trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Daily 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 200-period SMA and 20-period breakout levels
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma_200[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_14_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr_daily = atr_14_aligned[i]
        atr_weekly = atr_14_w_aligned[i]
        
        # Volatility filter: daily ATR < 1.5 * weekly ATR (avoid extremely high vol)
        vol_filter = atr_daily < (1.5 * atr_weekly)
        
        if position == 0:
            # Long: price breaks above 20-day high with uptrend and volatility filter
            if close[i] > high_20[i] and close[i] > sma_200[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below 20-day low with downtrend and volatility filter
            elif close[i] < low_20[i] and close[i] < sma_200[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 20-day low or trend turns down
            if close[i] < low_20[i] or close[i] < sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above 20-day high or trend turns up
            if close[i] > high_20[i] or close[i] > sma_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ATR_Filtered_Breakout_200SMA"
timeframe = "1d"
leverage = 1.0