#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ATR(14) for volatility calculation
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # True Range calculation
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 12h Donchian(20) channels
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Volatility filter: avoid low volatility (< 0.5x ATR) and extreme volatility (> 3.0x ATR)
        atr_ma = pd.Series(atr_14d_aligned).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma[i]):
            continue
        vol_filter = (atr_14d_aligned[i] > 0.5 * atr_ma[i]) and (atr_14d_aligned[i] < 3.0 * atr_ma[i])
        
        # Long: Price breaks above Donchian upper + volume spike + volatility filter
        if (close[i] > highest_high[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian lower + volume spike + volatility filter
        elif (close[i] < lowest_low[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price returns to the middle of the channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (highest_high[i] + lowest_low[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (highest_high[i] + lowest_low[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian20_Vol1.5x_ATR14Filter"
timeframe = "12h"
leverage = 1.0