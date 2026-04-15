#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. We use 1d EMA50 for trend direction
# and volume spikes to confirm momentum. Only take counter-trend pulls back in strong trends
# (buy pullbacks in uptrend, sell rallies in downtrend). This avoids whipsaw in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on 12h
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low + 1e-10)
    williams_r[highest_high == lowest_low] = -50  # Handle division by zero
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe (our primary timeframe)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: Williams %R crosses above -80 (oversold) in uptrend + volume spike
        if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
            close[i] > ema50_1d_aligned[i] and  # Uptrend filter: price above 1d EMA50
            volume[i] > 2.0 * vol_avg_aligned[i] and  # Volume spike confirmation
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R crosses below -20 (overbought) in downtrend + volume spike
        elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
              close[i] < ema50_1d_aligned[i] and  # Downtrend filter: price below 1d EMA50
              volume[i] > 2.0 * vol_avg_aligned[i] and  # Volume spike confirmation
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R crosses opposite threshold or trend reversal
        elif position == 1 and (williams_r_aligned[i] < -20 or close[i] < ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (williams_r_aligned[i] > -80 or close[i] > ema50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_Trend_Volume"
timeframe = "12h"
leverage = 1.0