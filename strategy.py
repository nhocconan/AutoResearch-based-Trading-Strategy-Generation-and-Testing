#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(34) trend filter and volume spike.
# Donchian breakout captures trend continuation. EMA(34) filter ensures we only trade in the direction of daily trend.
# Volume spike confirms institutional participation. Designed for ~15-25 trades/year per symbol.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 30-period average (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above Donchian upper band in uptrend with volume
        if close[i] > highest_high[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
            signals[i] = 0.25
            position = 1
        # Short: price breaks below Donchian lower band in downtrend with volume
        elif close[i] < lowest_low[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
            signals[i] = -0.25
            position = -1
        # Exit long: price crosses below EMA
        elif position == 1 and close[i] < ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Exit short: price crosses above EMA
        elif position == -1 and close[i] > ema34_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_DonchianBreakout_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0