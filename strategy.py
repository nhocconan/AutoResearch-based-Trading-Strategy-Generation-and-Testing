#!/usr/bin/env python3
"""
12h_WeeklyDonchian20_Breakout_1dTrend_RegimeFilter
Hypothesis: Trade 12h timeframe using weekly Donchian channel (20) breakout for entry,
daily EMA50 for trend filter, and daily choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend).
Enter long when price breaks above weekly Donchian high AND above daily EMA50 AND CHOP < 38.2 (trending).
Enter short when price breaks below weekly Donchian low AND below daily EMA50 AND CHOP < 38.2 (trending).
Exit on opposite Donchian touch or trend reversal. Uses discrete sizing 0.25 to balance return and drawdown.
Target 12-37 trades/year on 12h timeframe. Works in bull/bear via weekly structure and trend filter with regime avoidance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for weekly Donchian channel (20)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channel (20-period)
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe (completed weekly bar only)
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Get 1d data for daily EMA50 trend filter and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily choppiness index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(1) over 14) / (log10(highest_high - lowest_low) * sqrt(14)))
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high - lowest_low) * np.sqrt(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_raw = np.log10(np.sum(pd.Series(atr1).rolling(window=14, min_periods=14).sum().values) / chop_denom) * 100
    chop_1d = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values  # smooth
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (50), ATR (14), CHOP (14)
    start_idx = max(20, 50, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND above daily EMA50 AND trending regime
            long_setup = (close[i] > donchian_high_1w_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         is_trending
            # Short: price breaks below weekly Donchian low AND below daily EMA50 AND trending regime
            short_setup = (close[i] < donchian_low_1w_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches weekly Donchian low OR closes below daily EMA50
            if (close[i] <= donchian_low_1w_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly Donchian high OR closes above daily EMA50
            if (close[i] >= donchian_high_1w_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WeeklyDonchian20_Breakout_1dTrend_RegimeFilter"
timeframe = "12h"
leverage = 1.0