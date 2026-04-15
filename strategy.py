#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Bollinger Bands + Volume Filter
# Williams %R identifies overbought/oversold conditions on 4h timeframe.
# 1d Bollinger Bands provide trend context: price above upper band = bullish bias, below lower band = bearish bias.
# Volume confirmation requires > 1.5x 20-bar median volume to filter low-quality signals.
# Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day Bollinger Bands (20, 2)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    # Determine bias: 1 = bullish (price above upper band), -1 = bearish (price below lower band), 0 = neutral
    bias = np.where(close_1d > upper_band.values, 1, np.where(close_1d < lower_band.values, -1, 0))
    bias_aligned = align_htf_to_ltf(prices, df_1d, bias)
    
    # 4h Williams %R (14 periods)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(bias_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Williams %R oversold (< -80) and bullish bias (1d price above upper BB)
        if (williams_r[i] < -80 and 
            bias_aligned[i] == 1 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20) and bearish bias (1d price below lower BB)
        elif (williams_r[i] > -20 and 
              bias_aligned[i] == -1 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-50 to -50) or bias changes
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (williams_r[i] >= -50 or bias_aligned[i] != 1)) or
               (signals[i-1] == -0.25 and (williams_r[i] <= -50 or bias_aligned[i] != -1)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsR_1dBB_Volume"
timeframe = "4h"
leverage = 1.0