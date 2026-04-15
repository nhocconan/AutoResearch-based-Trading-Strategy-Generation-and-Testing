#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + 1w Trend (Weekly MA50) + Volume Confirmation
# Uses 1-day Donchian channel breakouts for entry, filtered by weekly trend direction (price above/below MA50).
# Volume confirmation requires > 1.5x 20-day median volume to avoid false breakouts.
# Designed to work in bull markets (breakouts with trend) and bear markets (avoid counter-trend breakouts).
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Donchian(20) channels
    def donchian_channels(high, low, window=20):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max()
        lower = pd.Series(low).rolling(window=window, min_periods=window).min()
        return upper.values, lower.values
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # 1-week MA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    ma50_1w_aligned = align_htf_to_ltf(prices, df_1w, ma50_1w.values)
    
    # Volume confirmation: current > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for MA50
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ma50_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian upper, price above weekly MA50, volume spike
        if (close[i] > dc_upper[i] and 
            close[i] > ma50_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian lower, price below weekly MA50, volume spike
        elif (close[i] < dc_lower[i] and 
              close[i] < ma50_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price returns to opposite Donchian band or volume drops
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < dc_lower[i]) or
               (signals[i-1] == -0.25 and close[i] > dc_upper[i]) or
               volume[i] <= vol_threshold[i])):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian20_1wMA50_Volume"
timeframe = "1d"
leverage = 1.0