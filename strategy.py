#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day volume regime filter
# Long when Williams %R(14) crosses above -20 AND daily volume > 1.5x 20-day average
# Short when Williams %R(14) crosses below -80 AND daily volume > 1.5x 20-day average
# Exit when Williams %R returns to opposite threshold (-80 for longs, -20 for shorts)
# Williams %R identifies overbought/oversold conditions; volume regime ensures trades occur during
# high-participation moves, reducing whipsaw in low-volume environments. Effective in both
# trending and ranging markets by capturing mean reversion within institutional interest zones.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 6h (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate daily volume average for regime filter (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (14 for Williams %R + buffer)
    start = 20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_1d_now = vol_avg_1d_aligned[i]
        vol_threshold = vol_avg_1d_aligned[i] * 1.5 if not np.isnan(vol_avg_1d_aligned[i]) else np.inf
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 (from below) + volume regime
            if i > start and williams_r[i-1] <= -20 and wr > -20 and vol_1d_now > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 (from above) + volume regime
            elif i > start and williams_r[i-1] >= -80 and wr < -80 and vol_1d_now > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -80 or below
            if wr <= -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -20 or above
            if wr >= -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsR_1dVolumeRegime"
timeframe = "6h"
leverage = 1.0