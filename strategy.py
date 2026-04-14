#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h High-Low Range Breakout with 1d Trend Filter
# Uses 6h high-low range expansion as momentum signal - breakouts from consolidation
# 1d EMA (50) provides trend filter to trade in direction of higher timeframe trend
# Works in both bull/bear markets by capturing momentum in trending direction
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h ATR (14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h range (high-low) and its 20-period average
    hl_range = high - low
    avg_range = pd.Series(hl_range).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for range average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(avg_range[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        current_range = hl_range[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: range expansion (>1.5x average) with upward bias in uptrend
            if current_range > 1.5 * avg_range[i] and close[i] > open_[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: range expansion (>1.5x average) with downward bias in downtrend
            elif current_range > 1.5 * avg_range[i] and close[i] < open_[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: range contraction or trend reversal
            if current_range < 0.8 * avg_range[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: range contraction or trend reversal
            if current_range < 0.8 * avg_range[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_RangeBreakout_1dEMA_Trend"
timeframe = "6h"
leverage = 1.0