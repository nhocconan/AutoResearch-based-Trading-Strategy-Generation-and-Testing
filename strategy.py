#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume spike confirmation.
# Elder Ray measures bull/bear power as (close - EMA13) and (high - EMA13) for bulls,
# and (low - EMA13) for bears. Trades in direction of 12h EMA34 trend with volume confirmation.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buy bull power dips in uptrend) and bear markets (sell bear power rallies in downtrend).
name = "6h_ElderRay_12hEMA34_VolumeSpike"
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
    
    # Get 12h data for EMA34 trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA13 for Elder Ray (using close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
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
            # Long: bull power positive AND above 12h EMA34 (uptrend) AND volume spike
            long_condition = (bull_power[i] > 0) and (close[i] > ema34_12h_aligned[i]) and vol_spike
            if long_condition:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative AND below 12h EMA34 (downtrend) AND volume spike
            elif (bear_power[i] < 0) and (close[i] < ema34_12h_aligned[i]) and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative OR price crosses below 12h EMA34
            exit_condition = (bull_power[i] <= 0) or (close[i] <= ema34_12h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive OR price crosses above 12h EMA34
            exit_condition = (bear_power[i] >= 0) or (close[i] >= ema34_12h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals