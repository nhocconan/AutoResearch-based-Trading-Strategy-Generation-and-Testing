#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA200 trend filter and 6h volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND 6h volume > 1.8x 20-period average AND price > 1d EMA200.
# Short when Williams %R crosses below -80 (overbought rejection) AND 6h volume > 1.8x 20-period average AND price < 1d EMA200.
# Exit when Williams %R crosses back below -50 (for long) or above -50 (for short) to capture mean reversion in ranging markets.
# Uses Williams %R for mean reversion in 6b timeframe with trend filter to avoid counter-trend trades.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_WilliamsR_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 6h volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20, volume spike, above 1d EMA200
            long_cond = (williams_r[i] > -20) and (williams_r[i-1] <= -20) and volume_filter[i] and (close[i] > ema200_1d_aligned[i])
            # Short conditions: Williams %R crosses below -80, volume spike, below 1d EMA200
            short_cond = (williams_r[i] < -80) and (williams_r[i-1] >= -80) and volume_filter[i] and (close[i] < ema200_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -50 (mean reversion signal)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back above -50 (mean reversion signal)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals