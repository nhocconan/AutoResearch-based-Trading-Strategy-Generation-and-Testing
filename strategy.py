#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Bollinger Band squeeze breakout with volume confirmation
# Long when 1d close breaks above upper BB AND 1w BB width at 20-period low AND volume > 1.5 * avg_volume(20)
# Short when 1d close breaks below lower BB AND 1w BB width at 20-period low AND volume > 1.5 * avg_volume(20)
# Exit when price crosses middle BB (20 SMA)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Bollinger Band squeeze identifies low volatility primed for breakout
# Weekly timeframe ensures we only trade breaks aligned with higher timeframe structure
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "1d_1wBB_Squeeze_Breakout_VolumeConfirm"
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
    
    # Calculate 1w Bollinger Bands (20, 2)
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / sma_20_1w  # Normalized width
    
    # Calculate 1w BB width percentile (20-period lookback for squeeze)
    bb_width_percentile = pd.Series(bb_width_1w).rolling(window=20, min_periods=20).rank(pct=True).values
    is_squeeze = bb_width_percentile <= 0.2  # Bottom 20% = squeeze
    
    # Align 1w indicators to 1d timeframe (wait for completed 1w bar)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    is_squeeze_aligned = align_htf_to_ltf(prices, df_1w, is_squeeze.astype(float))
    
    # Calculate 1d Bollinger Bands for exit (middle band = 20 SMA)
    sma_20_1d = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
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
            np.isnan(lower_bb_1w_aligned[i]) or np.isnan(is_squeeze_aligned[i]) or 
            np.isnan(sma_20_1d[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d close breaks above upper BB AND 1w BB squeeze AND volume spike, in session
            if (close[i] > upper_bb_1w_aligned[i] and 
                is_squeeze_aligned[i] > 0.5 and  # Boolean as float: 1.0 = True
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d close breaks below lower BB AND 1w BB squeeze AND volume spike, in session
            elif (close[i] < lower_bb_1w_aligned[i] and 
                  is_squeeze_aligned[i] > 0.5 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle BB (20 SMA)
            if close[i] < sma_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle BB (20 SMA)
            if close[i] > sma_20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals