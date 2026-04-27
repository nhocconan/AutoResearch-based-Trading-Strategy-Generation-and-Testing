#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly EMA200 trend filter and volume confirmation.
# Long when price breaks above 20-day high with weekly uptrend and volume spike.
# Short when price breaks below 20-day low with weekly downtrend and volume spike.
# Uses daily timeframe for structure, weekly for trend to avoid whipsaw.
# Volume filter ensures entry on expanding participation.
# Designed for 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate 200-period EMA on weekly close
    ema200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Calculate 20-day Donchian channels on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_weekly_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 20-day high AND weekly uptrend AND volume spike
        if (close[i] > high_20[i] and 
            close[i] > ema200_weekly_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below 20-day low AND weekly downtrend AND volume spike
        elif (close[i] < low_20[i] and 
              close[i] < ema200_weekly_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA200_VolumeFilter"
timeframe = "1d"
leverage = 1.0