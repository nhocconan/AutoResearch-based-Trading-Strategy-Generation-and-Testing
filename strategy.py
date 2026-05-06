#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Bollinger Band squeeze + 1d Williams %R mean reversion + volume confirmation
# Long when Bollinger Band Width (20,2) < 20th percentile (squeeze) AND Williams %R < -80 (oversold) AND volume > 1.5 * avg_volume(20) on 6h
# Short when Bollinger Band Width (20,2) < 20th percentile (squeeze) AND Williams %R > -20 (overbought) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when Williams %R crosses back through -50 (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Bollinger Band squeeze identifies low volatility periods primed for breakout
# Williams %R extremes provide high-probability reversal points in ranging markets
# Volume confirmation validates breakout strength while limiting overtrading
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "6h_1dBB_Squeeze_1dWilliamsR_MeanRev_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for Bollinger Bands and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Bollinger Bands
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 1d Bollinger Band Width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values * 100
    bb_squeeze = bb_width_percentile < 20  # Squeeze when width < 20th percentile
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze, Williams %R < -80 (oversold), volume spike, in session
            if (bb_squeeze_aligned[i] and 
                williams_r_aligned[i] < -80 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze, Williams %R > -20 (overbought), volume spike, in session
            elif (bb_squeeze_aligned[i] and 
                  williams_r_aligned[i] > -20 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals