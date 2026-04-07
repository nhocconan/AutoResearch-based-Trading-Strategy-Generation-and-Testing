#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout + 1d EMA(50) Trend + Volume Confirmation
# Hypothesis: Breakouts from 20-period price channels in direction of daily trend
# with volume confirmation work in both bull and bear markets by capturing
# momentum bursts while avoiding false breakouts. Target: 80-180 total trades over 4 years (20-45/year).

name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian Channel (20-period high/low)
    donch_period = 20
    highest_20 = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lowest_20 = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1e-10)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donch_period, n):
        # Skip if required data not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period low or trend changes
            if close[i] < lowest_20[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 20-period high or trend changes
            if close[i] > highest_20[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation threshold
            vol_confirmed = vol_ratio[i] > 1.5
            
            # Breakout above 20-period high with uptrend and volume
            if close[i] > highest_20[i] and close[i] > ema_50_1d_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Breakdown below 20-period low with downtrend and volume
            elif close[i] < lowest_20[i] and close[i] < ema_50_1d_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals