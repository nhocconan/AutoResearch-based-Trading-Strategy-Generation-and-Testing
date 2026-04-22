#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R with 1-day trend filter and volume confirmation.
Enter long when Williams %R crosses above -80 (oversold) in a rising 1-day trend.
Enter short when Williams %R crosses below -20 (overbought) in a falling 1-day trend.
Exit when Williams %R returns to the -50 level (mean reversion) or trend changes.
Williams %R identifies exhaustion points; 1-day trend filters for higher timeframe direction.
Designed for low trade frequency by requiring oversold/overbought conditions + trend alignment.
Works in both bull and bear markets by trading pullbacks in the direction of the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = ((highest_high - close) / hl_range) * -100.0
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) in uptrend with volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # Rising trend
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) in downtrend with volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # Falling trend
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns to -50 (mean reversion) OR trend turns down
                if (williams_r[i] >= -50 or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R returns to -50 (mean reversion) OR trend turns up
                if (williams_r[i] <= -50 or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0