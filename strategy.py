#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R crosses above -20 from below, daily EMA(34) uptrend, and volume spike
# Short when Williams %R crosses below -80 from above, daily EMA(34) downtrend, and volume spike
# Williams %R identifies overbought/oversold conditions; crosses signal momentum shifts
# Daily EMA ensures alignment with higher timeframe trend
# Volume spike confirms institutional participation; reduces false signals
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Williams %R and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on daily data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Period 14
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R formula
    williams_r = -100 * (highest_high - daily_close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (available after daily close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -20, daily uptrend, volume spike
            # Need previous Williams %R value to detect cross
            if i > start_idx:
                prev_williams_r = williams_r_aligned[i-1]
                if (prev_williams_r <= -20 and williams_r_val > -20 and 
                    price > ema34_1d_val and vol_spike):
                    signals[i] = 0.25
                    position = 1
            # Enter short: Williams %R crosses below -80 from above, daily downtrend, volume spike
                elif (prev_williams_r >= -80 and williams_r_val < -80 and 
                      price < ema34_1d_val and vol_spike):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R falls below -80 or daily trend turns down
            if williams_r_val < -80 or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R rises above -20 or daily trend turns up
            if williams_r_val > -20 or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals