#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d EMA34 filter and volume spike.
# Elder Ray measures bull/bear power relative to EMA13, indicating trend strength.
# 1d EMA34 filter ensures trades align with higher timeframe trend.
# Volume spike confirms momentum behind the move.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (bull power > 0) and bear markets (bear power < 0).
name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Calculate daily EMA34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periodas=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume above 1.5x average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: bull power positive AND price above daily EMA34 AND volume spike
            long_condition = bull_power[i] > 0 and close[i] > ema34_1d_aligned[i] and vol_spike
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative AND price below daily EMA34 AND volume spike
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative OR price crosses below daily EMA34
            exit_condition = bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive OR price crosses above daily EMA34
            exit_condition = bear_power[i] >= 0 or close[i] > ema34_1d_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals