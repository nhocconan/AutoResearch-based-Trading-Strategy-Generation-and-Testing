#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend_Filter
Hypothesis: Weekly pivot points (R2/S2) on daily chart provide strong support/resistance.
Breakouts above R2 or below S2 with volume confirmation and trend alignment (using 200-day EMA)
capture significant moves in both bull and bear markets. Uses weekly pivot calculation for
higher timeframe structure. Designed for low-frequency, high-quality trades (target: 10-20/year)
to minimize fee drag while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 250:  # Need sufficient data for 200 EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Use previous week's high, low, close
    prev_week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(5).values
    prev_week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(5).values
    prev_week_close = df_1d['close'].rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot point (P)
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    # Weekly R2 and S2
    R2 = pivot + (prev_week_high - prev_week_low)
    S2 = pivot - (prev_week_high - prev_week_low)
    
    # 200-day EMA for trend filter
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly pivot data to daily
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 250  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA200 = bullish bias, < EMA200 = bearish bias
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Entry conditions
        # Long: price breaks above R2 + bullish bias + volume surge
        long_entry = (close[i] > R2_aligned[i] and 
                     bullish_bias and 
                     volume_surge[i])
        
        # Short: price breaks below S2 + bearish bias + volume surge
        short_entry = (close[i] < S2_aligned[i] and 
                      bearish_bias and 
                      volume_surge[i])
        
        # Exit on opposite pivot level
        long_exit = close[i] < S2_aligned[i]
        short_exit = close[i] > R2_aligned[i]
        
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

name = "1d_WeeklyPivot_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0