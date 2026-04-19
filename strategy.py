#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h_1d_1w_TrendFollowing_Breakout
# Hypothesis: Use 12h timeframe with 1d EMA trend filter and 1w ATR volatility filter.
# Enter long when price breaks above 12h Donchian(20) high with 1d EMA200 filter and 1w volatility expansion.
# Enter short when price breaks below 12h Donchian(20) low with 1d EMA200 filter and 1w volatility expansion.
# Uses volatility expansion to avoid whipsaws in ranging markets. Designed for low trade frequency (<30/year).
# Works in both bull and bear markets by following trend with volatility confirmation.
name = "12h_1d_1w_TrendFollowing_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1w ATR for volatility filter (expanding volatility signal)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, 
                       np.absolute(high_1w - np.roll(close_1w, 1)), 
                       np.absolute(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1w = atr_1w / atr_1w_ma_50  # Current ATR vs 50-period average
    atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200, 50)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_1w_aligned[i]) or \
           np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: only trade when volatility is expanding (above average)
        vol_expansion = atr_ratio_1w_aligned[i] > 1.1
        
        if position == 0:
            # Long: price breaks above Donchian high + above EMA200 + volatility expansion
            if price > donch_high[i] and price > ema200_1d_aligned[i] and vol_expansion:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below EMA200 + volatility expansion
            elif price < donch_low[i] and price < ema200_1d_aligned[i] and vol_expansion:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or drops below EMA200
            if price < donch_low[i] or price < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or rises above EMA200
            if price > donch_high[i] or price > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals