#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Williams %R extremes with 1d EMA50 trend filter and volume confirmation
# Long when weekly %R < -80 (oversold) AND price > 1d EMA50 AND 6h volume > 1.3 * avg_volume(20)
# Short when weekly %R > -20 (overbought) AND price < 1d EMA50 AND 6h volume > 1.3 * avg_volume(20)
# Exit when weekly %R returns to -50 level or opposite extreme
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 1d EMA50 ensures we trade with the intermediate trend while reducing noise
# Volume confirmation filters out low-conviction reversal signals
# Works in bull markets (buying oversold dips) and bear markets (selling overbought rallies)

name = "6h_1wWilliamsR_Extreme_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 completed weekly bars for Williams %R
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14) * -100
    
    # Align weekly Williams %R to 6h timeframe (wait for completed 1w bar)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly %R oversold (< -80), price above 1d EMA50, volume spike
            if (williams_r_1w_aligned[i] < -80 and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly %R overbought (> -20), price below 1d EMA50, volume spike
            elif (williams_r_1w_aligned[i] > -20 and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly %R returns to -50 level or goes overbought
            if williams_r_1w_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly %R returns to -50 level or goes oversold
            if williams_r_1w_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals