#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R reversal with weekly trend filter and volume confirmation
# We go long when Williams %R crosses above -80 from oversold with weekly EMA(34) uptrend and volume spike.
# We go short when Williams %R crosses below -20 from overbought with weekly EMA(34) downtrend and volume spike.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the reversal.
# Target: 12-37 trades/year on 6h timeframe to avoid excessive frequency.

name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Get weekly data once for Williams %R and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - weekly_close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close_ewm = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w = weekly_close_ewm
    
    # Align Williams %R and weekly EMA to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Previous Williams %R values for crossover detection
        wr_prev = williams_r_aligned[i-1]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from oversold + weekly uptrend + volume spike
            if (wr_prev <= -80 and wr > -80 and 
                close[i] > ema34_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from overbought + weekly downtrend + volume spike
            elif (wr_prev >= -20 and wr < -20 and 
                  close[i] < ema34_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 OR weekly trend turns down
            if (wr_prev >= -50 and wr < -50) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 OR weekly trend turns up
            if (wr_prev <= -50 and wr > -50) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals