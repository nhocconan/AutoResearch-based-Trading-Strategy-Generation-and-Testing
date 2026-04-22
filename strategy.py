#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear power) with 1d trend filter and volume confirmation.
# Uses daily EMA13 for trend, and Elder Ray = (High - EMA13) for bull power, (Low - EMA13) for bear power.
# Goes long when bull power > 0 and bear power < 0 in uptrend with volume spike.
# Short when bear power < 0 and bull power > 0 in downtrend with volume spike.
# Session filter (08-20 UTC) reduces noise. Target: 12-30 trades/year (48-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend and Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # High - EMA13
    bear_power = low_1d - ema_13_1d   # Low - EMA13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power > 0, Bear power < 0, Uptrend, Volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema_13_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power < 0, Bull power > 0, Downtrend, Volume spike
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] > 0 and 
                  close[i] < ema_13_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: When Elder Ray signals weaken or trend changes
            if position == 1:
                if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or close[i] <= ema_13_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0 or close[i] >= ema_13_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_ElderRay_1dTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0