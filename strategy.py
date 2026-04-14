#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes + volume confirmation
# - Long when 1d Williams %R < -80 (oversold) and price breaks above 6h EMA20 with volume > 1.5x average
# - Short when 1d Williams %R > -20 (overbought) and price breaks below 6h EMA20 with volume > 1.5x average
# - Williams %R identifies overextended conditions; EMA20 provides dynamic trend filter
# - Volume confirmation ensures institutional participation at reversal points
# - Works in both bull and bear markets by fading extremes
# - Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate 6h EMA20
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(ema20[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get 1d index for current 6h bar (4 bars per day)
        idx_1d = i // 4
        if idx_1d < 1:
            continue
            
        # Previous day's Williams %R (to avoid look-ahead)
        wr_prev = williams_r[idx_1d-1]
        
        if position == 0:
            # Long: Oversold (Williams %R < -80) + price above EMA20 + volume confirmation
            if (wr_prev < -80 and  # Oversold condition from previous day
                close[i] > ema20[i] and  # Price above EMA20
                volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short: Overbought (Williams %R > -20) + price below EMA20 + volume confirmation
            elif (wr_prev > -20 and  # Overbought condition from previous day
                  close[i] < ema20[i] and  # Price below EMA20
                  volume[i] > vol_ma[i] * 1.5):  # Volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below EMA20 or Williams %R exits oversold
            if close[i] < ema20[i] or wr_prev > -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above EMA20 or Williams %R exits overbought
            if close[i] > ema20[i] or wr_prev < -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1d_WilliamsR_EMA20_Volume"
timeframe = "6h"
leverage = 1.0