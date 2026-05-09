#!/usr/bin/env python3
# Hypothesis: 4h timeframe with daily Bollinger Band breakout and 1w trend filter.
# Uses daily Bollinger Bands (20,2) for mean-reversion entries and weekly EMA50 for trend filter.
# In bull markets, buy dips below lower BB in uptrend; in bear markets, sell rallies above upper BB in downtrend.
# Weekly trend filter reduces whipsaw by aligning with higher timeframe direction.
# Target: 80-180 total trades over 4 years (20-45/year) with size 0.25.

name = "4h_Bollinger_MeanReversion_1wEMA50_Trend"
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
    
    # Calculate daily Bollinger Bands (20,2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_lower = bb_middle - 2 * bb_std
    bb_upper = bb_middle + 2 * bb_std
    
    # Mean-reversion conditions: price touches or crosses Bollinger Bands
    bb_lower_touch = close <= bb_lower
    bb_upper_touch = close >= bb_upper
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume (moderate to avoid overtrading)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_lower_touch[i]) or np.isnan(bb_upper_touch[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/below lower BB + weekly uptrend + volume filter
            if bb_lower_touch[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches/above upper BB + weekly downtrend + volume filter
            elif bb_upper_touch[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB or trend reversal
            if close[i] >= bb_middle[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB or trend reversal
            if close[i] <= bb_middle[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals