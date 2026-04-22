#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
# Long when Bull Power > 0 and Bear Power < 0 (bullish market structure) with 1d EMA50 uptrend and volume spike.
# Short when Bear Power > 0 and Bull Power < 0 (bearish market structure) with 1d EMA50 downtrend and volume spike.
# Works in bull/bear markets by using 1d EMA50 trend filter to align with higher timeframe direction.
# Session filter (08-20 UTC) reduces noise. Target: 12-37 trades/year (50-150 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Bull Power > 0 and Bear Power < 0 (bullish structure) + 1d uptrend + volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 and Bull Power < 0 (bearish structure) + 1d downtrend + volume spike
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray signals weaken or reverse
            if position == 1:
                if bull_power[i] <= 0 or bear_power[i] >= 0:  # Bullish structure broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power[i] <= 0 or bull_power[i] >= 0:  # Bearish structure broken
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_ElderRay_1dTrend_Volume_Session"
timeframe = "6h"
leverage = 1.0