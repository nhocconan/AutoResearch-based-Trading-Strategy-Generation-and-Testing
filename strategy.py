#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R Mean Reversion with 1-day Trend Filter
# Williams %R identifies overbought/oversold conditions; 1-day EMA200 filters for higher timeframe trend
# Only takes counter-trend reversals when price is overextended against the daily trend
# Designed for 4h timeframe with selective entries to avoid overtrading
# Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-period EMA on 1d timeframe for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Williams %R on 4h timeframe (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema200_1d_aligned[i]
        is_downtrend = close[i] < ema200_1d_aligned[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) AND price above 1d EMA200 (contrarian to short-term)
            # Actually wait: in uptrend, oversold is buy; in downtrend, overbought is sell
            if williams_r[i] < -80 and is_uptrend:
                signals[i] = 0.25
                position = 1
            elif williams_r[i] > -20 and is_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or trend breaks
            if williams_r[i] > -50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or trend breaks
            if williams_r[i] < -50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA200_TrendFilter"
timeframe = "4h"
leverage = 1.0