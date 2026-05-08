#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -20 from below (oversold reversal) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 from above (overbought reversal) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back through -50 (mean reversion to midpoint).
# Williams %R captures momentum reversals. EMA50 filters higher timeframe trend.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsR_Reversal_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    
    # Williams %R signals: -20 (overbought threshold), -80 (oversold threshold), -50 (midline exit)
    williams_r_overbought = -20
    williams_r_oversold = -80
    williams_r_midline = -50
    
    # Align Williams %R levels to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Sufficient warmup for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20 from below, EMA50 rising, volume filter
            williams_r_cross_up = (williams_r_aligned[i] > williams_r_overbought) and (williams_r_aligned[i-1] <= williams_r_overbought)
            long_cond = williams_r_cross_up and ema50_rising[i] and volume_filter[i]
            
            # Short conditions: Williams %R crosses below -80 from above, EMA50 falling, volume filter
            williams_r_cross_down = (williams_r_aligned[i] < williams_r_oversold) and (williams_r_aligned[i-1] >= williams_r_oversold)
            short_cond = williams_r_cross_down and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -50 (midline) from above
            williams_r_cross_down_mid = (williams_r_aligned[i] < williams_r_midline) and (williams_r_aligned[i-1] >= williams_r_midline)
            if williams_r_cross_down_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back above -50 (midline) from below
            williams_r_cross_up_mid = (williams_r_aligned[i] > williams_r_midline) and (williams_r_aligned[i-1] <= williams_r_midline)
            if williams_r_cross_up_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals