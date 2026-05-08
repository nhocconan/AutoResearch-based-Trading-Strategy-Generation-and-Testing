#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# Long when price breaks above 4h Donchian upper channel (20) + 1d volume > 1.5x 20-period average + 1w EMA(34) up
# Short when price breaks below 4h Donchian lower channel (20) + 1d volume > 1.5x 20-period average + 1w EMA(34) down
# Exit when price crosses back through Donchian middle (10-period average of high/low)
# Uses volume to confirm breakout strength and weekly trend to avoid counter-trend trades
# Targets 20-50 trades per year to minimize fee drag

name = "4h_DonchianBreakout_1dVolume_1wTrend"
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
    
    # Get 1d data once for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    
    # 1w EMA(34) for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2  # Middle for exit
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_20_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_20_val = vol_20_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: break above Donchian high + volume confirmation + weekly uptrend
            if close[i] > donch_high[i] and volume[i] > 1.5 * vol_20_val and ema34_1w_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + volume confirmation + weekly downtrend
            elif close[i] < donch_low[i] and volume[i] > 1.5 * vol_20_val and ema34_1w_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals