# 4h_Donchian20_VolumeSpike_TrendFilter
# Strategy type: Donchian(20) breakout with volume spike and 1d trend filter for 4h timeframe
# Rationale: Donchian(20) captures price breakouts; volume spike confirms momentum; 1d EMA50 filters trend direction. Works in bull/bear by following 1d trend direction. Volume filter ensures momentum; 4h balances trade frequency and signal quality.
# Target: 20-40 trades/year per symbol with strict entry conditions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade only during active hours (8-20 UTC) to avoid low-volume periods
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above Donchian high, 1d uptrend (price > EMA50), volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, 1d downtrend (price < EMA50), volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if close[i] < low_20[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if close[i] > high_20[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals