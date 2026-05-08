#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with daily trend filter and volume confirmation
# We go long when Williams %R crosses above -20 (oversold) with daily EMA(34) uptrend and volume spike.
# We go short when Williams %R crosses below -80 (overbought) with daily EMA(34) downtrend and volume spike.
# Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
# In trending markets, it can signal pullbacks in the direction of the trend.
# The daily trend filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation in the move.
# Target: 12-37 trades/year on 6h timeframe to avoid excessive frequency and fee drag.

name = "6h_WilliamsR_DailyTrend_Volume"
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
    
    # Get daily data once for Williams %R calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate rolling highest high and lowest low for Williams %R
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    williams_r = np.where(range_hl != 0, (highest_high - daily_close) / range_hl * -100, -50)
    
    # Williams %R signals: above -20 is overbought, below -80 is oversold
    # We'll use crosses of these levels for entries
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
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
        
        wr = williams_r_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (from oversold) + daily uptrend + volume spike
            # We need to check if previous value was below -80 and current is above -80
            if i > start_idx:
                prev_wr = williams_r_aligned[i-1]
                if (not np.isnan(prev_wr) and prev_wr < -80 and wr >= -80 and 
                    close[i] > ema34_1d_val and vol_spike):
                    signals[i] = 0.25
                    position = 1
            # Enter short: Williams %R crosses below -20 (from overbought) + daily downtrend + volume spike
                elif (not np.isnan(prev_wr) and prev_wr > -20 and wr <= -20 and 
                      close[i] < ema34_1d_val and vol_spike):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR daily trend turns down
            if i > start_idx:
                prev_wr = williams_r_aligned[i-1]
                if (not np.isnan(prev_wr) and prev_wr < -20 and wr >= -20) or close[i] < ema34_1d_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR daily trend turns up
            if i > start_idx:
                prev_wr = williams_r_aligned[i-1]
                if (not np.isnan(prev_wr) and prev_wr > -80 and wr <= -80) or close[i] > ema34_1d_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals