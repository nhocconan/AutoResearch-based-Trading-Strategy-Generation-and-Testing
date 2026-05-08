#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h EMA200 trend filter and 4h volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND 4h volume > 1.8x 20-period average AND price > 4h EMA200.
# Short when Williams %R crosses below -80 (overbought rejection) AND 4h volume > 1.8x 20-period average AND price < 4h EMA200.
# Exit when Williams %R crosses back below -50 (for long) or above -50 (for short).
# Uses 4h for signal direction/trend/volume, 1h only for entry timing precision.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h to minimize fee drag.

name = "1h_WilliamsR_4hEMA200_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = prices.index.hour
    
    # 4h data for EMA trend filter and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 4h volume filter: current volume > 1.8x 20-period average
    vol_ma20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_4h = df_4h['volume'].values
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    vol_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20_4h)
    volume_filter = vol_4h_aligned > (1.8 * vol_ma20_4h_aligned)
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema200_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20, volume spike, above 4h EMA200, in session
            long_cond = (williams_r[i] > -20) and (williams_r[i-1] <= -20) and volume_filter[i] and (close[i] > ema200_4h_aligned[i]) and in_session
            # Short conditions: Williams %R crosses below -80, volume spike, below 4h EMA200, in session
            short_cond = (williams_r[i] < -80) and (williams_r[i-1] >= -80) and volume_filter[i] and (close[i] < ema200_4h_aligned[i]) and in_session
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -50 (mean reversion signal)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Williams %R crosses back above -50 (mean reversion signal)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals