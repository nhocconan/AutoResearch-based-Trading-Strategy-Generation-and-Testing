#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA200 filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1d EMA200 AND volume > 2.0x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1d EMA200 AND volume > 2.0x 20-period average.
Exit when Williams %R returns to opposite extreme (-20 for longs, -80 for shorts).
Designed to capture mean reversion in 6h timeframe with institutional volume confirmation.
Works in both bull and bear markets by fading extremes with trend filter.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA200 on 1d close
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema200 = ema200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND price > 1d EMA200 AND volume spike
            if i > start_idx and williams_r_aligned[i-1] <= -80 and wr > -80 and price > ema200 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND price < 1d EMA200 AND volume spike
            elif i > start_idx and williams_r_aligned[i-1] >= -20 and wr < -20 and price < ema200 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -20 (overbought territory)
            if wr >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -80 (oversold territory)
            if wr <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0