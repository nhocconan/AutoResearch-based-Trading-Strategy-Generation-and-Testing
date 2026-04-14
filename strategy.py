#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1-week trend filter
# Uses Williams %R (%R) for mean reversion signals on daily timeframe
# In oversold (%R < -80) go long, overbought (%R > -20) go short
# Weekly EMA50 acts as trend filter: only take long when price > weekly EMA50, short when price < weekly EMA50
# Works in both bull/bear by combining mean reversion with trend alignment
# Target: 30-100 total trades over 4 years (7-25/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R (14-period) on daily data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.where(hh_ll != 0, -100 * (highest_high - close) / hh_ll, -50)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: oversold (%R < -80) and price above weekly EMA50 (uptrend)
            if williams_r[i] < -80 and price > ema_50_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: overbought (%R > -20) and price below weekly EMA50 (downtrend)
            elif williams_r[i] > -20 and price < ema_50_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: either overbought (%R > -20) or price breaks below weekly EMA50
            if williams_r[i] > -20 or price < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: either oversold (%R < -80) or price breaks above weekly EMA50
            if williams_r[i] < -80 or price > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WilliamsR_WeeklyEMA_Filter"
timeframe = "1d"
leverage = 1.0