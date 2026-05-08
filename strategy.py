#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Volume Spike and 1d Trend Filter
# Long when price breaks above Camarilla R3 AND volume > 2x 20-period average AND price > 1d EMA50
# Short when price breaks below Camarilla S3 AND volume > 2x 20-period average AND price < 1d EMA50
# Exit when price returns to Camarilla Pivot Point (PP)
# Uses Camarilla levels from daily timeframe for institutional reference points
# Target: 80-120 total trades over 4 years (20-30/year) for low fee drift

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Volume"
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
    
    # 1d data for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculation: PP = (H+L+C)/3, Range = H-L
    pp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + (rng * 1.1 / 2.0)  # Resistance 3
    s3 = pp - (rng * 1.1 / 2.0)  # Support 3
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, above 1d EMA50
            long_cond = (close[i] > r3_aligned[i]) and (close[i-1] <= r3_aligned[i-1]) and volume_filter[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: price breaks below S3, volume spike, below 1d EMA50
            short_cond = (close[i] < s3_aligned[i]) and (close[i-1] >= s3_aligned[i-1]) and volume_filter[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point (mean reversion)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot point (mean reversion)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals