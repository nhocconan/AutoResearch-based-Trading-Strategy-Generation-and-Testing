#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions
# Long: Williams %R < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume > 1.5x avg
# Short: Williams %R > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume > 1.5x avg
# Uses 12h EMA50 for trend filter to avoid counter-trend trades
# Williams %R period: 14 (standard)
# Volume filter: current volume > 1.5x 20-period average
# Target: 20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R < -80 (oversold) AND price above 12h EMA50 (uptrend) AND volume confirmation
        if (williams_r[i] < -80 and 
            close[i] > ema50_12h_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R > -20 (overbought) AND price below 12h EMA50 (downtrend) AND volume confirmation
        elif (williams_r[i] > -20 and 
              close[i] < ema50_12h_aligned[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0