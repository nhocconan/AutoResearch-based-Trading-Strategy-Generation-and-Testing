#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R with Bollinger Band squeeze filter
# Long when %R < -80 (oversold) AND BB width < 20th percentile (low volatility squeeze)
# Short when %R > -20 (overbought) AND BB width < 20th percentile
# Exit when %R returns to -50 (mean reversion midpoint)
# Williams %R identifies overextended moves; Bollinger squeeze identifies low volatility
# periods where mean reversion is more likely to succeed. Works in both bull/bear markets
# by fading extremes during consolidation phases.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for BB(20)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate 20th percentile of BB width for squeeze filter
    # Use expanding window to avoid look-ahead
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.expanding(min_periods=20).quantile(0.20).values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # Need BB(20) period
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: current BB width < 20th percentile (low volatility)
        # Need current BB width - calculate it for current bar
        if i >= 20:  # Need enough data for current BB width
            current_bb_middle = np.nanmean(close_1d[max(0, i-16):i+1])  # Approximate for alignment
            current_bb_std = np.nanstd(close_1d[max(0, i-16):i+1]) if i >= 16 else 0
            current_bb_width = 4 * current_bb_std  # 2 * std * 2 (upper - lower)
        else:
            current_bb_width = np.inf  # No squeeze early on
        
        squeeze_condition = current_bb_width < bb_width_percentile_aligned[i]
        
        if position == 0:
            # Look for mean reversion entries during low volatility squeeze
            # Long: %R < -80 (oversold) AND squeeze
            if (williams_r_aligned[i] < -80 and 
                squeeze_condition):
                position = 1
                signals[i] = position_size
            # Short: %R > -20 (overbought) AND squeeze
            elif (williams_r_aligned[i] > -20 and 
                  squeeze_condition):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: %R returns to -50 (mean reversion midpoint)
            if williams_r_aligned[i] >= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: %R returns to -50
            if williams_r_aligned[i] <= -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsR_BB_Squeeze_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0