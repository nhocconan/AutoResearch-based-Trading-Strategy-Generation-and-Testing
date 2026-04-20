#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with daily volume confirmation and trend filter.
# Williams %R < -80 = oversold (long), > -20 = overbought (short).
# Uses 1d EMA50 as trend filter to avoid counter-trend trades.
# Volume > 1.5x 20-period average confirms momentum.
# Designed for fewer trades (<40/year) to avoid fee drag, works in bull/bear via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Williams %R, volume, and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Daily volume average for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: oversold + above EMA50 + volume confirmation
            if (williams_r_aligned[i] < -80 and 
                price > ema_50_1d_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought + below EMA50 + volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  price < ema_50_1d_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 or trend changes
            if williams_r_aligned[i] > -50 or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 or trend changes
            if williams_r_aligned[i] < -50 or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0