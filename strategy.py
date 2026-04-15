#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w SMA50 trend filter and volume confirmation
# Donchian breakout captures strong trends, 1w SMA50 filters for major trend alignment,
# volume spike (>1.5x 20-bar median) ensures institutional participation.
# Designed for low-frequency, high-conviction trades to avoid fee drag.
# Works in bull markets (breakouts) and bear markets (breakdowns) with trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels on 1d: 20-period high/low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1w SMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above Donchian high + above 1w SMA50 + volume spike
        if (close[i] > donch_high[i] and 
            close[i] > sma_50_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low + below 1w SMA50 + volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < sma_50_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Price crosses back through Donchian middle or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and 
                (close[i] < (donch_high[i] + donch_low[i]) / 2 or 
                 close[i] < sma_50_1w_aligned[i])) or
               (signals[i-1] == -0.25 and 
                (close[i] > (donch_high[i] + donch_low[i]) / 2 or 
                 close[i] > sma_50_1w_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Donchian20_1wSMA50_Volume"
timeframe = "1d"
leverage = 1.0