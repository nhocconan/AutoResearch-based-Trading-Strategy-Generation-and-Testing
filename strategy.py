#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # 1d daily high-low range for volatility filter
    df_1d = get_htf_data(prices, '1d')
    hl_range = df_1d['high'].values - df_1d['low'].values
    range_ma = pd.Series(hl_range).rolling(window=20, min_periods=20).mean().values
    range_ma_aligned = align_htf_to_ltf(prices, df_1d, range_ma)
    
    # 12h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current > 2.0x median of last 50 periods
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median().values
    
    # Volatility filter: require current daily range > 0.5x its MA
    vol_filter = hl_range > 0.5 * range_ma_aligned
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_median[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: price breaks above Donchian high + volume + volatility filter
        if close[i] > donch_high[i] and volume[i] > 2.0 * vol_median[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume + volatility filter
        elif close[i] < donch_low[i] and volume[i] > 2.0 * vol_median[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: price returns inside Donchian channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_VolFilter"
timeframe = "12h"
leverage = 1.0