#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 20-period Donchian breakout with daily trend filter (EMA34) and volume confirmation
# We go long when price breaks above the 20-day high with daily EMA(34) uptrend and volume spike.
# We go short when price breaks below the 20-day low with daily EMA(34) downtrend and volume spike.
# Designed for low trade frequency in both bull and bear markets with proper risk control.
# Target: 20-50 total trades over 4 years = 5-12/year (well within limits)

name = "4h_Donchian20_1dTrend_Volume"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 20-period Donchian channels on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + daily EMA34 uptrend + volume spike
            if (close[i] > upper_band and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + daily EMA34 downtrend + volume spike
            elif (close[i] < lower_band and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR daily EMA34 turns down
            if (close[i] < lower_band or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR daily EMA34 turns up
            if (close[i] > upper_band or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals