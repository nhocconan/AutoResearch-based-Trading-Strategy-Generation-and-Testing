#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1w trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions (%R < -80 = oversold, > -20 = overbought)
# Long: %R < -80 (oversold) and price above weekly EMA34 (uptrend) and volume > 1.5x 20-period average
# Short: %R > -20 (overbought) and price below weekly EMA34 (downtrend) and volume > 1.5x 20-period average
# Uses 1w EMA34 for trend filter to avoid whipsaws in sideways markets
# Target: 12-30 trades/year per symbol (48-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period Williams %R on weekly data
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low)
    
    # 34-period EMA on weekly close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Williams %R < -80 (oversold) AND price above weekly EMA34 (uptrend) AND volume confirmation
        if (williams_r[i] < -80 and 
            close[i] > ema34_1w_aligned[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Williams %R > -20 (overbought) AND price below weekly EMA34 (downtrend) AND volume confirmation
        elif (williams_r[i] > -20 and 
              close[i] < ema34_1w_aligned[i] and volume_filter[i]):
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

name = "12h_WilliamsR_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0