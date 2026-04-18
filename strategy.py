#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 filter and volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA, capturing trend strength.
# Daily EMA34 filter ensures alignment with higher timeframe trend.
# Volume spike confirms institutional participation.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (strong bull power) and bear markets (strong bear power).
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
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate daily EMA34 for trend filter
    close_d = df_1d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 6h timeframe
    ema34_d_aligned = align_htf_to_ltf(prices, df_1d, ema34_d)
    
    # Calculate 24-period average volume for spike detection (approx 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_std_24 = pd.Series(volume).rolling(window=24, min_periods=24).std().values
    vol_threshold = vol_ma_24 + 2.0 * vol_std_24  # Volume spike = 2 std above mean
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume[i] > vol_threshold[i]
        
        if position == 0:
            # Long: strong bull power, price above daily EMA34, volume spike
            if bull_power[i] > 0 and close[i] > ema34_d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: strong bear power, price below daily EMA34, volume spike
            elif bear_power[i] < 0 and close[i] < ema34_d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bear power becomes positive (bulls losing control) OR volume dries up
            exit_condition = bear_power[i] > 0 or volume[i] < vol_ma_24[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bull power becomes negative (bears losing control) OR volume dries up
            exit_condition = bull_power[i] < 0 or volume[i] < vol_ma_24[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals