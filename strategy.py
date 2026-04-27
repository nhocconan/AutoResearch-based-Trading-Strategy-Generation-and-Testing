#!/usr/bin/env python3
"""
12h_Williams_R_Reversal_1dTrend_Filter
Hypothesis: Williams %R overbought/oversold reversals on 12h timeframe filtered by daily trend.
In bull markets: buy oversold pullbacks in uptrend. In bear markets: sell overbought bounces in downtrend.
Williams %R identifies exhaustion points; daily trend filter ensures alignment with higher timeframe momentum.
Target: 15-30 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Williams %R and EMA50
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) and price above daily EMA50 (uptrend)
            if wr < -80 and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) and price below daily EMA50 (downtrend)
            elif wr > -20 and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend turns down
            if wr > -50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend turns up
            if wr < -50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_R_Reversal_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0