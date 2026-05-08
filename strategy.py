#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour 20-period Donchian breakout with 12-hour EMA trend and volume confirmation
# We go long when price breaks above the 4h Donchian high with 12h EMA(21) uptrend and volume spike.
# We go short when price breaks below the 4h Donchian low with 12h EMA(21) downtrend and volume spike.
# The 12h EMA provides a medium-term trend filter to avoid counter-trend trades.
# Volume spike confirms breakout strength. Designed for low trade frequency (target: 20-50/year).
# Works in bull markets via breakouts and in bear markets via short breakdowns.

name = "4h_Donchian20_12hTrend_Volume"
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
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA(21) for trend direction
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate 4h 20-period Donchian channels
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
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema21_12h_val = ema21_12h_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + 12h uptrend + volume spike
            if (close[i] > upper_band and 
                close[i] > ema21_12h_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + 12h downtrend + volume spike
            elif (close[i] < lower_band and 
                  close[i] < ema21_12h_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 12h trend turns down
            if (close[i] < lower_band or close[i] < ema21_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 12h trend turns up
            if (close[i] > upper_band or close[i] > ema21_12h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals