#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d Bollinger Band squeeze filter and volume confirmation.
Long when Williams %R crosses above -80 from below AND BB width < 20th percentile AND volume > 1.5x average.
Short when Williams %R crosses below -20 from above AND BB width < 20th percentile AND volume > 1.5x average.
Exit when Williams %R reverses or BB width expands above 50th percentile.
Williams %R identifies oversold/overbought conditions for mean reversion in ranging markets.
BB squeeze filter ensures low volatility environment where mean reversion works best.
Volume confirmation avoids false signals. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in both bull and bear markets by focusing on mean reversion during low volatility regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Band squeeze filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands on 1d data (20, 2)
    bb_ma_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper_1d = bb_ma_1d + 2 * bb_std_1d
    bb_lower_1d = bb_ma_1d - 2 * bb_std_1d
    bb_width_1d = (bb_upper_1d - bb_lower_1d) / bb_ma_1d
    
    # Align 1d BB width to 6h timeframe
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate BB width percentiles (20th and 50th) using expanding window
    bb_width_series = pd.Series(bb_width_1d)
    bb_width_p20 = bb_width_series.expanding(min_periods=50).quantile(0.20).values
    bb_width_p50 = bb_width_series.expanding(min_periods=50).quantile(0.50).values
    bb_width_p20_aligned = align_htf_to_ltf(prices, df_1d, bb_width_p20)
    bb_width_p50_aligned = align_htf_to_ltf(prices, df_1d, bb_width_p50)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bb_width_1d_aligned[i]) or np.isnan(bb_width_p20_aligned[i]) or 
            np.isnan(bb_width_p50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = williams_r[i] if not np.isnan(williams_r[i]) else prev_williams_r
            continue
        
        bb_width_val = bb_width_1d_aligned[i]
        bb_width_p20_val = bb_width_p20_aligned[i]
        bb_width_p50_val = bb_width_p50_aligned[i]
        williams_r_val = williams_r[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND BB squeeze AND volume spike
            crossed_up = (prev_williams_r <= -80 and williams_r_val > -80)
            bb_squeeze = bb_width_val < bb_width_p20_val
            volume_spike = vol_current > 1.5 * vol_ma_val
            
            if crossed_up and bb_squeeze and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND BB squeeze AND volume spike
            elif (prev_williams_r >= -20 and williams_r_val < -20) and bb_squeeze and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 OR BB width expands above 50th percentile
                if (prev_williams_r < -20 and williams_r_val >= -20) or bb_width_val > bb_width_p50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -80 OR BB width expands above 50th percentile
                if (prev_williams_r > -80 and williams_r_val <= -80) or bb_width_val > bb_width_p50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
        
        prev_williams_r = williams_r_val
    
    return signals

name = "6H_WilliamsR_1dBBSqueeze_Volume"
timeframe = "6h"
leverage = 1.0