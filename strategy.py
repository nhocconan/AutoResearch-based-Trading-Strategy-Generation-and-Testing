# 6h_Donchian20_WeeklyTrend_VolumeFilter
# Hypothesis: 6h breakout of 20-period Donchian channel with weekly trend filter and volume confirmation.
# Uses 1-week trend direction (EMA50) to determine bias, Donchian breakout for entry, and volume spike to avoid false signals.
# Weekly trend filter provides stability in both bull and bear markets by aligning with higher timeframe structure.
# Target: 70-140 total trades over 4 years = 17-35/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly trend to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly EMA, Donchian channels, and volume data
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (volume spike)
        vol_filter = vol_current > (vol_ma_val * 2.0)
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly uptrend and volume spike
            if close[i] > high_max[i] and close[i] > weekly_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with weekly downtrend and volume spike
            elif close[i] < low_min[i] and close[i] < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian low or weekly trend turns down
            if close[i] < low_min[i] or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian high or weekly trend turns up
            if close[i] > high_max[i] or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0