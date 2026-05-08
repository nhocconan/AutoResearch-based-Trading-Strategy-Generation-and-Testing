#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high in uptrend (price > EMA200) with volume spike.
# Short when price breaks below Donchian(20) low in downtrend (price < EMA200) with volume spike.
# Exit when price returns to Donchian midpoint (mean of high/low over 20 periods).
# Uses Donchian channels from 4h data, EMA200 from daily for trend filter, volume > 2x 20-period average.
# Designed to capture trend continuation moves with volume confirmation in both bull and bear markets.
# Target: 20-50 trades/year to minimize fee decay while maintaining edge.

name = "4h_Donchian_Breakout_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian channel (20-period high/low) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Calculate 20-period average volume for volume filter (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + trend + volume
            # Long when price breaks above Donchian high in uptrend (price > EMA200) with volume spike
            long_condition = (high[i] > high_roll[i]) and \
                           (close[i] > ema_200_aligned[i]) and vol_filter
            # Short when price breaks below Donchian low in downtrend (price < EMA200) with volume spike
            short_condition = (low[i] < low_roll[i]) and \
                            (close[i] < ema_200_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals