#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA200 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 4h EMA200 provides strong trend filter to avoid counter-trend trades in both bull/bear markets
# Volume confirmation (>1.5x 20 EMA) ensures breakout/participation validity
# Session filter (08-20 UTC) focuses on high-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: EMA200 ensures we only trade with the major trend,
# Williams %R provides precise mean-reversion entries during pullbacks.

name = "1h_WilliamsR_4hEMA200_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200
    close_4h = df_4h['close'].values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMA200 to 1h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Williams %R (14-period) on 1h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) + price above 4h EMA200 + volume spike
            if williams_r[i] < -80 and close[i] > ema_200_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: Williams %R overbought (> -20) + price below 4h EMA200 + volume spike
            elif williams_r[i] > -20 and close[i] < ema_200_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) OR price crosses below 4h EMA200
            if williams_r[i] > -50 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) OR price crosses above 4h EMA200
            if williams_r[i] < -50 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals