#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation
# Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Oversold: < -80, Overbought: > -20
# Trend filter: 1w EMA50 to avoid counter-trend trades in strong trends
# Volume > 1.5x 20-period average confirms reversal conviction
# Target: 10-20 trades/year per symbol (40-80 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # 50-period EMA on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R oversold (< -80) AND price above weekly EMA50 (uptrend filter)
        # AND volume confirmation
        if (williams_r[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R overbought (> -20) AND price below weekly EMA50 (downtrend filter)
        # AND volume confirmation
        elif (williams_r[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_filter[i]):
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

name = "1d_WilliamsR_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0