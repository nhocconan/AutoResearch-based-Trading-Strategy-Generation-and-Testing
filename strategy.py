#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R mean reversion with weekly trend filter and volume confirmation
# We go long when Williams %R < -80 (oversold) with weekly EMA(50) uptrend and volume spike.
# We go short when Williams %R > -20 (overbought) with weekly EMA(50) downtrend and volume spike.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the reversal.
# Target: 15-25 trades/year to avoid fee drag while capturing mean reversion edges.

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - weekly_close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close_series = pd.Series(weekly_close)
    ema50_1w = weekly_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align Williams %R and EMA to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + weekly uptrend + volume spike
            if wr < -80 and close[i] > ema50 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + weekly downtrend + volume spike
            elif wr > -20 and close[i] < ema50 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) OR weekly trend turns down
            if wr > -50 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) OR weekly trend turns up
            if wr < -50 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals