#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume confirmation and 1w trend filter.
# Works in bull/bear by filtering breaks with weekly trend (avoid counter-trend).
# Target: 20-50 trades over 4 years to minimize fee drag.
name = "1d_Donchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # 1w EMA200 for trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(ema200_1w_aligned[i])):
            continue
        
        # Long: break above Donchian high + volume + price above weekly EMA200
        if close[i] > donch_high[i] and volume[i] > vol_threshold[i] and close[i] > ema200_1w_aligned[i]:
            signals[i] = 0.25
        
        # Short: break below Donchian low + volume + price below weekly EMA200
        elif close[i] < donch_low[i] and volume[i] > vol_threshold[i] and close[i] < ema200_1w_aligned[i]:
            signals[i] = -0.25
        
        # Exit: price crosses back inside Donchian channels (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals