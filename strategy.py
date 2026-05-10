#!/usr/bin/env python3
# 6H_Williams_R_Extremes_DailyTrend_Filter
# Hypothesis: Uses 6h timeframe with Williams %R to identify extreme oversold/overbought conditions.
# Enters long when Williams %R crosses above -80 from below (oversold bounce) AND price > 1d EMA200 (bullish bias).
# Enters short when Williams %R crosses below -20 from above (overbought rejection) AND price < 1d EMA200 (bearish bias).
# Uses daily EMA200 for trend to avoid counter-trend trades and works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25 to minimize fee drag.

name = "6H_Williams_R_Extremes_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Williams %R and EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 6h timeframe (available after 1d bar closes)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA(200) for trend direction
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    start_idx = max(200, 14)  # Warmup for EMA200 and Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(williams_r_aligned[i]) or np.isnan(williams_r_aligned[i-1]) or np.isnan(ema_200_1d_aligned[i]):
            continue
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Williams %R conditions
        wr_current = williams_r_aligned[i]
        wr_previous = williams_r_aligned[i-1]
        
        # Long: Williams %R crosses above -80 from below (oversold bounce) in uptrend
        if (wr_previous <= -80 and wr_current > -80 and price_above_ema):
            signals[i] = 0.25
        # Short: Williams %R crosses below -20 from above (overbought rejection) in downtrend
        elif (wr_previous >= -20 and wr_current < -20 and price_below_ema):
            signals[i] = -0.25
    
    return signals