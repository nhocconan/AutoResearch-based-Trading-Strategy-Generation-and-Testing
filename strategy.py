#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter and volume spike
# Williams %R identifies overbought/oversold conditions. Works in both bull and bear:
# - Bull market: Buy when Williams %R crosses above -80 from below + price > 200-day SMA
# - Bear market: Sell when Williams %R crosses below -20 from above + price < 200-day SMA
# Volume spike filters weak moves. 1w trend filter ensures alignment with higher timeframe momentum.
# Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1w SMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    sma50_1w = close_1w.rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 from below + price above 1w SMA50 + volume
        if (williams_r_aligned[i] > -80 and 
            williams_r_aligned[i-1] <= -80 and  # crossed above -80
            close[i] > sma50_1w_aligned[i] and   # uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 from above + price below 1w SMA50 + volume
        elif (williams_r_aligned[i] < -20 and 
              williams_r_aligned[i-1] >= -20 and  # crossed below -20
              close[i] < sma50_1w_aligned[i] and   # downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "1d_WilliamsR_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0