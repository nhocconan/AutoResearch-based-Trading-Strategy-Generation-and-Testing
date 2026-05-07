#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action filtered by 1d Bollinger Band squeeze and volume expansion.
# Long when price closes above upper Bollinger Band with volume > 1.5x 20-period average.
# Short when price closes below lower Bollinger Band with volume > 1.5x 20-period average.
# Uses Bollinger Band width < 50th percentile to identify low volatility squeeze conditions.
# Designed for low frequency (target: 10-20 trades/year) to minimize fee drag.
# Works in ranging markets by capturing volatility expansion after contraction.
name = "12h_Bollinger_Squeeze_Volume_Expansion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    
    # 1d volume expansion: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_expansion = np.where(vol_ma_20 > 0, df_1d['volume'].values / vol_ma_20, 1.0) > 1.5
    
    # Align 1d indicators to 12h timeframe
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_expansion_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: price above upper BB, BB squeeze (width < 50th percentile), volume expansion
            long_condition = (close[i] > bb_upper_aligned[i]) and \
                           (bb_width_percentile_aligned[i] < 50.0) and \
                           vol_expansion_aligned[i]
            # Short condition: price below lower BB, BB squeeze (width < 50th percentile), volume expansion
            short_condition = (close[i] < bb_lower_aligned[i]) and \
                            (bb_width_percentile_aligned[i] < 50.0) and \
                            vol_expansion_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below middle BB or volatility expansion ends
            bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
            if (close[i] < bb_middle_aligned[i]) or not vol_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above middle BB or volatility expansion ends
            bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
            if (close[i] > bb_middle_aligned[i]) or not vol_expansion_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals