#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray + Weekly Trend Filter with Volume Confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising + weekly EMA(34) uptrend + volume spike
# Short when Bear Power < 0 and falling + weekly EMA(34) downtrend + volume spike
# Uses weekly timeframe for trend filter to reduce whipsaw in sideways markets
# Volume spike confirms institutional participation
# Targets 12-37 trades/year on 6h timeframe to avoid fee drag

name = "6h_ElderRay_WeeklyTrend_Volume"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA(13)
    bear_power = low - ema13   # Low - EMA(13)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend = ema34_1w_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising + weekly uptrend + volume spike
            if (bull > 0 and i > start_idx and bull > bull_power[i-1] and 
                close[i] > weekly_trend and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 and falling + weekly downtrend + volume spike
            elif (bear < 0 and i > start_idx and bear < bear_power[i-1] and 
                  close[i] < weekly_trend and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 or weekly trend turns down
            if bull <= 0 or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 or weekly trend turns up
            if bear >= 0 or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals