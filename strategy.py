#!/usr/bin/env python3
# Hypothesis: 4h strategy using 1d Bollinger Bands squeeze (volatility contraction) as regime filter,
# combined with 4h price breaking above/below Bollinger Bands (20,2) for mean-reversion entries.
# Bollinger Squeeze identifies low volatility periods that often precede explosive moves.
# Mean-reversion at Bollinger Band extremes works well in ranging markets, which dominate 2025.
# Uses volume confirmation to avoid false breakouts. Target: 20-50 trades/year.

name = "4h_BollingerSqueeze_MR_1dVolFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # 1d Bollinger Band width for squeeze detection (volatility regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < bb_period:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'].values)
    bb_middle_1d = close_1d.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev_1d = close_1d.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper_1d = bb_middle_1d + (bb_std_dev_1d * bb_std)
    bb_lower_1d = bb_middle_1d - (bb_std_dev_1d * bb_std)
    bb_width_1d = bb_upper_1d - bb_lower_1d
    
    # Squeeze condition: 1d BB width below its 50-period mean (low volatility)
    bb_width_1d_mean = bb_width_1d.rolling(window=50, min_periods=50).mean()
    squeeze_condition = bb_width_1d < bb_width_1d_mean
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.values)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # Enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(squeeze_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean-reversion long: price at or below lower BB + squeeze + volume
            if close[i] <= bb_lower[i] and squeeze_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Mean-reversion short: price at or above upper BB + squeeze + volume
            elif close[i] >= bb_upper[i] and squeeze_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB or squeeze breaks
            if close[i] >= bb_middle[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB or squeeze breaks
            if close[i] <= bb_middle[i] or not squeeze_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals