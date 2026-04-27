#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike
# Williams %R measures momentum and identifies overbought/oversold conditions.
# Works in both bull and bear markets by fading extremes in trending markets.
# - In uptrend: Buy when %R crosses above -80 from oversold + price > EMA50
# - In downtrend: Sell when %R crosses below -20 from overbought + price < EMA50
# Volume spike confirms momentum strength. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Williams %R (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close_1d) / (highest_high - lowest_low))
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d EMA50 for trend filter
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 (oversold) in uptrend + volume
        if (williams_r_aligned[i] > -80 and 
            williams_r_aligned[i-1] <= -80 and  # Cross above -80
            close[i] > ema50_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 (overbought) in downtrend + volume
        elif (williams_r_aligned[i] < -20 and 
              williams_r_aligned[i-1] >= -20 and  # Cross below -20
              close[i] < ema50_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0