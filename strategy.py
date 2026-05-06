#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Bollinger Band squeeze + mean reversion with volume confirmation
# Long when price touches lower 1w BB (20,2) AND 1w BB width < 20th percentile (squeeze) AND volume > 1.5 * avg_volume(20) on 1d
# Short when price touches upper 1w BB (20,2) AND 1w BB width < 20th percentile (squeeze) AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price crosses 1w 20-period SMA (mean reversion to middle band)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Bollinger squeeze identifies low volatility primed for expansion
# Mean reversion from extremes works well in ranging markets (2025+ bear/range)
# Volume confirmation validates breakout strength while preventing false signals
# Works in both bull (buy lower band dips) and bear (sell upper band rallies)

name = "1d_1wBB_Squeeze_MeanReversion_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Bollinger Band calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for BB
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Bollinger Bands (20,2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + (2.0 * std_20_1w)
    lower_bb_1w = sma_20_1w - (2.0 * std_20_1w)
    
    # Calculate 1w BB width for squeeze detection: (Upper - Lower) / SMA
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / sma_20_1w
    # Handle division by zero
    bb_width_1w = np.where(sma_20_1w == 0, 0, bb_width_1w)
    
    # Calculate 20th percentile of BB width for squeeze threshold (using expanding window)
    bb_width_percentile_20 = pd.Series(bb_width_1w).expanding(min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width_1w < bb_width_percentile_20
    
    # Align 1w indicators to 1d timeframe (wait for completed 1w bar)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    squeeze_aligned = align_htf_to_ltf(prices, df_1w, squeeze_condition)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(sma_20_1w_aligned[i]) or np.isnan(upper_bb_1w_aligned[i]) or 
            np.isnan(lower_bb_1w_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower 1w BB, BB squeeze, volume spike, in session
            if (low[i] <= lower_bb_1w_aligned[i] and 
                squeeze_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches upper 1w BB, BB squeeze, volume spike, in session
            elif (high[i] >= upper_bb_1w_aligned[i] and 
                  squeeze_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above 1w 20-period SMA (mean reversion)
            if close[i] > sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below 1w 20-period SMA (mean reversion)
            if close[i] < sma_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals