#!/usr/bin/env python3
"""
4h_Keltner_Touch_WeeklyTrend_Volume
Hypothesis: 4-hour touches at Keltner Channel bands with weekly trend filter and volume confirmation. Targets 20-50 trades/year by requiring price touches at volatility-based bands with trend alignment and volume surge. Works in both bull (long at lower band in uptrend) and bear (short at upper band in downtrend) markets.
"""

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
    
    # Get 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate Keltner Channel (20, 2.0) from 4h data
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean()
    upper = ema_20 + (atr * 2.0)
    lower = ema_20 - (atr * 2.0)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 4h
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper.values)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower.values)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price touches lower band + weekly uptrend + volume surge
        long_entry = (close[i] <= lower_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price touches upper band + weekly downtrend + volume surge
        short_entry = (close[i] >= upper_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite band touch with volume surge
        long_exit = close[i] >= upper_aligned[i] and volume_surge[i]
        short_exit = close[i] <= lower_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Keltner_Touch_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0