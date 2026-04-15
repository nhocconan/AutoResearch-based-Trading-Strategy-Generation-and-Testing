#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Weekly Donchian breakout with volume confirmation and trend filter
# Hypothesis: Weekly breakouts capture major trends while daily timeframe
# avoids overtrading. Volume confirms breakout strength. Works in both
# bull and bear markets by following the dominant weekly trend.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period)
    high_1w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max()
    low_1w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min()
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align to daily timeframe
    upper_1w = align_htf_to_ltf(prices, df_1w, high_1w.values)
    lower_1w = align_htf_to_ltf(prices, df_1w, low_1w.values)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w.values)
    
    # Daily volume confirmation: current > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(upper_1w[i]) or np.isnan(lower_1w[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            continue
        
        # Long: price breaks above weekly upper band + above weekly EMA50 + volume confirmation
        if close[i] > upper_1w[i] and close[i] > ema50_1w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: price breaks below weekly lower band + below weekly EMA50 + volume confirmation
        elif close[i] < lower_1w[i] and close[i] < ema50_1w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: price crosses back to weekly EMA50 (trend exhaustion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema50_1w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema50_1w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian20_EMA50_Volume"
timeframe = "1d"
leverage = 1.0