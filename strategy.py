#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Donchian breakout with 4-hour trend filter, daily volume confirmation, and session filter (08-20 UTC)
# We go long when price breaks above 4-hour Donchian upper band with 4-hour EMA(50) uptrend, daily volume spike, and within active session.
# We go short when price breaks below 4-hour Donchian lower band with 4-hour EMA(50) downtrend, daily volume spike, and within active session.
# Uses 1h timeframe targeting 15-37 trades/year by using 4h for signal direction (reducing frequency) and 1h only for entry timing.
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# Session filter reduces noise during low-volume periods.
# Volume spike confirms institutional participation in the breakout.

name = "1h_DonchianBreakout_4hTrend_DailyVolume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4-hour data once for trend filter and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # already datetime64[ms], .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_4h_val = ema50_4h_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        vol_spike = volume[i] > (2.0 * vol_ma_1d_val)
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian upper + 4h EMA(50) uptrend + daily volume spike + session
            if (close[i] > donchian_high_val and close[i] > ema50_4h_val and 
                vol_spike and in_session):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian lower + 4h EMA(50) downtrend + daily volume spike + session
            elif (close[i] < donchian_low_val and close[i] < ema50_4h_val and 
                  vol_spike and in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian lower OR 4h EMA(50) turns down
            if (close[i] < donchian_low_val) or (close[i] < ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian upper OR 4h EMA(50) turns up
            if (close[i] > donchian_high_val) or (close[i] > ema50_4h_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals