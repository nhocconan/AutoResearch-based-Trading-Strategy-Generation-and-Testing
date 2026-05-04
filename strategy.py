#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. Combined with 1d EMA50 trend alignment
# and volume spike (2.0x 20-period EMA), this filters for high-probability breakouts in
# the direction of the daily trend. Designed for 4h timeframe to target 20-50 trades/year
# (75-200 total over 4 years) with discrete sizing (0.30). Works in bull markets by buying
# breakouts above upper channel in uptrends and in bear markets by selling breakdowns below
# lower channel in downtrends, avoiding counter-trend entries.

name = "4h_Donchian20_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + price above 1d EMA50 (uptrend)
            if (close[i] > highest_high[i] and volume_confirmed and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + price below 1d EMA50 (downtrend)
            elif (close[i] < lowest_low[i] and volume_confirmed and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel (below upper) OR price below 1d EMA50 (trend change)
            if close[i] < highest_high[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price re-enters Donchian channel (above lower) OR price above 1d EMA50 (trend change)
            if close[i] > lowest_low[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals