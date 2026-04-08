#!/usr/bin/env python3
# 6h_1d_3bar_volatility_breakout
# Hypothesis: Buy volatility expansion after 3-bar low volatility contractions on 6h, confirmed by 1d trend.
# Uses Bollinger Band width to detect low volatility (squeeze), then breaks above/below Bollinger Bands.
# 1d EMA50 provides trend filter: long only when above EMA50, short only when below.
# Works in both bull (trend continuation) and bear (mean reversion at extremes) markets.
# Target: 15-30 trades/year on 6h timeframe with strict volatility breakout conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_3bar_volatility_breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for volatility measurement
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bb_width = (upper - lower) / ma20  # Normalized bandwidth
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 3-bar low volatility condition: BB width below 20-period average for 3 consecutive bars
    bb_ma20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    low_vol = bb_width < bb_ma20
    low_vol_3bar = (
        low_vol & 
        np.roll(low_vol, 1) & 
        np.roll(low_vol, 2)
    )
    # Handle NaN from roll
    low_vol_3bar[:2] = False
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 20  # For Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if required data not available
        if np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long condition: volatility breakout upward with 1d uptrend
        if (low_vol_3bar[i-1] and  # Volatility contraction ended 1 bar ago
            close[i] > upper[i] and  # Break above upper Bollinger Band
            close[i-1] <= upper[i-1] and  # Was inside or below band previously
            close[i] > ema50_1d_aligned[i]):  # Above daily EMA50 (uptrend filter)
            signals[i] = 0.25
        
        # Short condition: volatility breakout downward with 1d downtrend
        elif (low_vol_3bar[i-1] and  # Volatility contraction ended 1 bar ago
              close[i] < lower[i] and  # Break below lower Bollinger Band
              close[i-1] >= lower[i-1] and  # Was inside or above band previously
              close[i] < ema50_1d_aligned[i]):  # Below daily EMA50 (downtrend filter)
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # Flat
    
    return signals