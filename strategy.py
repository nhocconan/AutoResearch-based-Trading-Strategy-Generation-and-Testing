#!/usr/bin/env python3
# 6H_MultiFactor_Trend_With_Volume_Confirmation
# Hypothesis: Combines 12h EMA50 trend filter with 6h price action patterns (higher highs/lows) and volume confirmation.
# Uses higher timeframe trend (12h EMA50) to filter direction, reducing false signals in choppy markets.
# Requires price making higher highs/lows in direction of trend for entry, with volume > 1.5x 20-period average.
# Designed to work in both bull and bear markets by aligning with 12h trend and requiring momentum confirmation.
# Targets 12-30 trades per year on 6h timeframe with discrete position sizing (0.25) to minimize churn.

name = "6H_MultiFactor_Trend_With_Volume_Confirmation"
timeframe = "6h"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Higher highs/lows detection (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Momentum confirmation: making higher highs/lows
        making_higher_high = high[i] > highest_high[i-1] if i > 0 else False
        making_lower_low = low[i] < lowest_low[i-1] if i > 0 else False
        
        if position == 0:
            # Long entry: price above 12h EMA50 + making higher high + volume confirmation
            if (price_above_ema and 
                making_higher_high and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below 12h EMA50 + making lower low + volume confirmation
            elif (price_below_ema and 
                  making_lower_low and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 12h EMA50 or loses momentum (lower low)
            if (close[i] < ema_50_12h_aligned[i] or 
                low[i] < lowest_low[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 12h EMA50 or loses momentum (higher high)
            if (close[i] > ema_50_12h_aligned[i] or 
                high[i] > highest_high[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals