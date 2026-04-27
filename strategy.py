#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1-week trend filter and volume spike
# Williams %R identifies overbought/oversold conditions. Works in both bull and bear:
# - Bull market: Buy when Williams %R crosses above -80 from below (oversold bounce)
# - Bear market: Sell when Williams %R crosses below -20 from above (overbought rejection)
# 1-week trend filter ensures we trade with the higher timeframe trend.
# Volume spike confirms strong momentum behind the move.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max().values
    lowest_low = low_1d.rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Get 1 week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 from below (oversold bounce)
        # AND price above weekly EMA50 (uptrend filter) + volume spike
        if (williams_r[i] > -80 and 
            williams_r[i-1] <= -80 and  # Cross above -80
            close[i] > ema50_1w_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 from above (overbought rejection)
        # AND price below weekly EMA50 (downtrend filter) + volume spike
        elif (williams_r[i] < -20 and 
              williams_r[i-1] >= -20 and  # Cross below -20
              close[i] < ema50_1w_aligned[i] and   # Downtrend filter
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