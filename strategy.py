#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 12h trend filter and volume spike confirmation
# Williams %R measures overbought/oversold levels. Works in both bull and bear markets:
# - Bull market: Buy when Williams %R crosses above -80 from oversold + price > 12h EMA50
# - Bear market: Sell when Williams %R crosses below -20 from overbought + price < 12h EMA50
# Volume spike filters weak moves. Target: 25-40 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Williams %R calculation (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align Williams %R to lower timeframe (15m equivalent for 4h data)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R crosses above -80 (from oversold) + uptrend + volume
        if (williams_r_aligned[i] > -80 and 
            williams_r_aligned[i-1] <= -80 and  # Cross above -80
            close[i] > ema50_12h_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R crosses below -20 (from overbought) + downtrend + volume
        elif (williams_r_aligned[i] < -20 and 
              williams_r_aligned[i-1] >= -20 and  # Cross below -20
              close[i] < ema50_12h_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_WilliamsR_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0