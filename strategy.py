#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA34 trend filter and 12h volume confirmation.
# Long when price breaks above 12h Donchian upper (20-period high) AND 12h volume > 2.0x 12-period average AND price > 1d EMA34.
# Short when price breaks below 12h Donchian lower (20-period low) AND 12h volume > 2.0x 12-period average AND price < 1d EMA34.
# Exit when price closes back inside the Donchian channel (below upper for long, above lower for short).
# Uses Donchian breakout for trend capture, volume for confirmation, EMA34 for trend filter to avoid counter-trend trades.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume filter: current volume > 2.0x 12-period average
    vol_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (2.0 * vol_ma12)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume spike, above 1d EMA34
            long_cond = (close[i] > highest_high[i]) and volume_filter[i] and (close[i] > ema34_1d_aligned[i])
            # Short conditions: price breaks below Donchian lower, volume spike, below 1d EMA34
            short_cond = (close[i] < lowest_low[i]) and volume_filter[i] and (close[i] < ema34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian upper (mean reversion signal)
            if close[i] < highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian lower (mean reversion signal)
            if close[i] > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals