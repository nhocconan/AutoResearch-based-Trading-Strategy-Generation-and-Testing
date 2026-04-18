#!/usr/bin/env python3
"""
6h_WeeklyVolatilityBreakout_Bias
6h strategy using weekly volatility contraction/expansion with daily bias filter.
- Entry: Price breaks out of weekly ATR-based range with volume surge + daily EMA alignment
- Exit: Opposite breakout or volatility contraction signal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation) by using volatility regime filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for volatility-based range
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(5) for dynamic range
    tr1_w = np.maximum(high_1w, np.concatenate([[close_1w[0]], close_1w[:-1]])) - np.minimum(low_1w, np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr2_w = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3_w = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_5_w = pd.Series(tr_w).rolling(window=5, min_periods=5).mean().values
    
    # Weekly range: center ± 1.5 * ATR
    weekly_center = (high_1w + low_1w) / 2
    weekly_range_width = 1.5 * atr_5_w
    weekly_high_range = weekly_center + weekly_range_width
    weekly_low_range = weekly_center - weekly_range_width
    
    # Align weekly ranges to 6h
    weekly_high_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_range)
    weekly_low_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_range)
    
    # Get daily data for bias filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for bias filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_range_aligned[i]) or np.isnan(weekly_low_range_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bias conditions
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]  # Higher threshold for 6h
        
        # Breakout conditions from weekly volatility range
        breakout_up = close[i] > weekly_high_range_aligned[i]
        breakdown_down = close[i] < weekly_low_range_aligned[i]
        
        if position == 0:
            # Long: bullish bias + volume surge + breakout above weekly range
            if bullish_bias and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish bias + volume surge + breakdown below weekly range
            elif bearish_bias and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish bias, volume surge with breakdown, or volatility contraction
            if bearish_bias or (vol_confirm and breakdown_down):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish bias, volume surge with breakout, or volatility contraction
            if bullish_bias or (vol_confirm and breakout_up):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyVolatilityBreakout_Bias"
timeframe = "6h"
leverage = 1.0