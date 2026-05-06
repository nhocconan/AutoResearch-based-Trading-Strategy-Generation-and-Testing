#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (> -20 for short, < -80 for long) 
# combined with weekly trend alignment reduces whipsaw in both bull/bear markets
# Volume spike (>1.3x 20-bar average) confirms momentum exhaustion
# Discrete sizing 0.25 to limit fee drag; target 60-120 total trades over 4 years (15-30/year)
# Williams %R is a proven mean reversion oscillator that works across market regimes when filtered by higher timeframe trend

name = "6h_WilliamsR_MeanRev_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R (14-period) on 1d timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero when high == low
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Calculate volume confirmation (>1.3x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) AND uptrend (price > weekly EMA50) AND volume confirmation
            if williams_r_1d_aligned[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) AND downtrend (price < weekly EMA50) AND volume confirmation
            elif williams_r_1d_aligned[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or weekly trend breaks
            if williams_r_1d_aligned[i] > -50 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or weekly trend breaks
            if williams_r_1d_aligned[i] < -50 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals