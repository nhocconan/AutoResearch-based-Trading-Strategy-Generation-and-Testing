#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion with 12h Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# Trades counter-trend when %R reaches extreme levels (>80 for oversold, <20 for overbought)
# Only takes trades in direction of 12h trend (using EMA crossover) to avoid fighting major trends
# Requires volume spike confirmation to ensure momentum behind the move
# Designed to work in both ranging and trending markets by combining mean reversion with trend filter
# Target: 60-120 trades over 4 years (15-30/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h EMA crossover for trend filter (9 and 21)
    close_12h = df_12h['close'].values
    ema_9 = pd.Series(close_12h).ewm(span=9, adjust=False).mean().values
    ema_21 = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    # 1 = uptrend (EMA9 > EMA21), -1 = downtrend (EMA9 < EMA21)
    trend = np.where(ema_9 > ema_21, 1, -1)
    
    # Calculate 12h volume average (20-period) for spike detection
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    trend_aligned = align_htf_to_ltf(prices, df_12h, trend)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Williams %R and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_12h_current = volume_12h[i] if i < len(volume_12h) else volume_12h[-1]
        
        if position == 0:
            # Long setup: Williams %R oversold (< -80) in uptrend with volume spike
            if (williams_r_aligned[i] < -80 and 
                trend_aligned[i] == 1 and 
                vol_12h_current > 1.5 * vol_ma_12h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R overbought (> -20) in downtrend with volume spike
            elif (williams_r_aligned[i] > -20 and 
                  trend_aligned[i] == -1 and 
                  vol_12h_current > 1.5 * vol_ma_12h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend changes
            if williams_r_aligned[i] > -50 or trend_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend changes
            if williams_r_aligned[i] < -50 or trend_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsR_MeanReversion_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0