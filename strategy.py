#!/usr/bin/env python3
name = "6h_Trend_Reversal_With_Liquidity_Sweep"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Daily ATR(14) for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 6h ATR(10) for liquidity sweep detection
    tr_6h = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                  np.abs(low[1:] - close[:-1])))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    atr_10_6h = pd.Series(tr_6h).rolling(window=10, min_periods=10).mean().values
    
    # Swing high/low detection on 6h (lookback 5 periods)
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    for i in range(5, n):
        if high[i] == np.max(high[i-5:i+1]):
            swing_high[i] = high[i]
        if low[i] == np.min(low[i-5:i+1]):
            swing_low[i] = low[i]
    
    # Liquidity sweep detection: price breaks swing level but reverses quickly
    bullish_sweep = np.zeros(n, dtype=bool)
    bearish_sweep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        # Bullish sweep: breaks swing low then closes above it
        if swing_low[i] != 0 and low[i] < swing_low[i] and close[i] > swing_low[i]:
            bullish_sweep[i] = True
        # Bearish sweep: breaks swing high then closes below it
        if swing_high[i] != 0 and high[i] > swing_high[i] and close[i] < swing_high[i]:
            bearish_sweep[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_100_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_10_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish liquidity sweep + above daily EMA100 + volatility filter
            if (bullish_sweep[i] and 
                close[i] > ema_100_1d_aligned[i] and 
                atr_10_6h[i] > 0.5 * atr_14_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish liquidity sweep + below daily EMA100 + volatility filter
            elif (bearish_sweep[i] and 
                  close[i] < ema_100_1d_aligned[i] and 
                  atr_10_6h[i] > 0.5 * atr_14_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish liquidity sweep or below EMA100
            if bearish_sweep[i] or close[i] < ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish liquidity sweep or above EMA100
            if bullish_sweep[i] or close[i] > ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals