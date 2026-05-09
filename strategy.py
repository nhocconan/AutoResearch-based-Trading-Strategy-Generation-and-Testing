#!/usr/bin/env python3
name = "6H_PairTrading_BTC_ETH_Zscore_1dVWAP"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (typical price * volume cumulative)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price_1d * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = np.divide(vwap_numerator, vwap_denominator, 
                        out=np.full_like(typical_price_1d, np.nan), 
                        where=vwap_denominator!=0)
    
    # Align daily VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate price deviation from VWAP (as percentage)
    price_dev = (close - vwap_1d_aligned) / vwap_1d_aligned * 100
    
    # Calculate rolling z-score of price deviation (20-day lookback)
    # Using pandas for efficient rolling z-score
    price_dev_series = pd.Series(price_dev)
    rolling_mean = price_dev_series.rolling(window=20, min_periods=20).mean().values
    rolling_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    zscore = np.divide(price_dev - rolling_mean, rolling_std, 
                       out=np.full_like(price_dev, np.nan), 
                       where=rolling_std!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for z-score
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if z-score not ready
        if np.isnan(zscore[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        z = zscore[i]
        
        if position == 0:
            # Enter long when price significantly below VWAP (mean reversion long)
            if z < -1.5:
                signals[i] = 0.25
                position = 1
            # Enter short when price significantly above VWAP (mean reversion short)
            elif z > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price reverts to VWAP or overshoots slightly
            if z > -0.5:  # Return to near VWAP or slight overshoot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price reverts to VWAP or undershoots slightly
            if z < 0.5:  # Return to near VWAP or slight undershoot
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals