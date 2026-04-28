#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Uses Williams %R(14) to identify oversold/overbought conditions. Enters long when %R crosses above -80
# in uptrend (price > 1d EMA34) with volume > 1.5x 20-period average. Shorts when %R crosses below -20
# in downtrend with volume confirmation. Exits when %R crosses -50 (mean reversion midpoint).
# Designed for 12h timeframe with ~50-150 total trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 12h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 20)  # Wait for EMA, Williams %R, and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Williams %R conditions
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r[i]
        
        # Entry conditions: Williams %R crosses oversold/overbought levels in trend direction with volume
        long_entry = (williams_r_prev <= -80 and williams_r[i] > -80 and uptrend and volume_confirm[i])
        short_entry = (williams_r_prev >= -20 and williams_r[i] < -20 and downtrend and volume_confirm[i])
        
        # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
        long_exit = (williams_r_prev < -50 and williams_r[i] >= -50) or (not uptrend)
        short_exit = (williams_r_prev > -50 and williams_r[i] <= -50) or (not downtrend)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0