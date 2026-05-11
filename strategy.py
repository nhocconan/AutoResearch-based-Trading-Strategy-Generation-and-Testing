#!/usr/bin/env python3
"""
6h_PriceAction_1dTrend_1wVolatility
Hypothesis: Combines 6h price action (breaking above/below 6h ATR-based channel) with 1d trend filter (price above/below 1d SMA200) and 1w volatility filter (low ATR percentile). 
This structure avoids whipsaws in ranging markets by requiring both trend alignment and low volatility for entries, while allowing exits on trend reversals. 
Designed for low trade frequency (<30/year) with strong performance in both bull and bear markets by following higher timeframe trends during calm periods.
"""

name = "6h_PriceAction_1dTrend_1wVolatility"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h ATR for channel calculation (using 14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h ATR-based channel (donchian-like with ATR bands)
    upper_channel = np.roll(close, 1) + atr
    lower_channel = np.roll(close, 1) - atr
    
    # 1d trend filter (SMA200)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # 1w volatility filter (ATR percentile - low volatility environment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    tr1w = df_1w['high'] - df_1w['low']
    tr2w = np.abs(df_1w['high'] - np.roll(df_1w['close'].values, 1))
    tr3w = np.abs(df_1w['low'] - np.roll(df_1w['close'].values, 1))
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_w = pd.Series(trw).rolling(window=14, min_periods=14).mean().values
    # Calculate 50-period percentile of ATR (low volatility = bottom 30%)
    atr_w_percentile = pd.Series(atr_w).rolling(window=50, min_periods=20).quantile(0.3).values
    atr_w_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_w_percentile)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(sma_200_1d_aligned[i]) or np.isnan(atr_w_percentile_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: current 6h ATR below 1w ATR 30th percentile (low volatility)
        low_volatility = atr[i] < atr_w_percentile_aligned[i]
        
        if position == 0:
            # Long: price breaks above 6h channel + above 1d SMA200 + low volatility
            if (close[i] > upper_channel[i] and 
                close[i] > sma_200_1d_aligned[i] and 
                low_volatility):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h channel + below 1d SMA200 + low volatility
            elif (close[i] < lower_channel[i] and 
                  close[i] < sma_200_1d_aligned[i] and 
                  low_volatility):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or volatility expansion
            if position == 1:
                # Exit long: price below 1d SMA200 OR volatility expands (ATR > 70th percentile)
                if (close[i] < sma_200_1d_aligned[i]) or \
                   (atr[i] > atr_w_percentile_aligned[i] * 2.0):  # volatility expansion
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price above 1d SMA200 OR volatility expands
                if (close[i] > sma_200_1d_aligned[i]) or \
                   (atr[i] > atr_w_percentile_aligned[i] * 2.0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals