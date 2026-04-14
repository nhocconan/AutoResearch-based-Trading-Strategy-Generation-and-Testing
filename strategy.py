#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Overbought/Oversold with 1d Trend Filter
# Uses Williams %R (14-period) to identify overextended conditions on 12h timeframe
# 1d EMA(50) acts as trend filter - only take counter-trend entries when price is
# extended against the daily trend. Works in both bull/bear by fading extremes
# in the direction of the higher timeframe trend. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R (14-period) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r).values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Williams %R and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) and price above 1d EMA (uptrend)
            # Fading the extreme in direction of trend
            if wr < -80 and price > ema_50:
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R overbought (> -20) and price below 1d EMA (downtrend)
            # Fading the extreme in direction of trend
            elif wr > -20 and price < ema_50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or reversal signal
            if wr > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or reversal signal
            if wr < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsR_1dEMA_Filter"
timeframe = "12h"
leverage = 1.0