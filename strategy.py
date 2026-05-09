#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe strategy using 4h Donchian breakout with volume confirmation and 1d trend filter.
# Strategy enters long when price breaks above 4h Donchian upper band with 1d uptrend and volume spike,
# short when price breaks below 4h Donchian lower band with 1d downtrend and volume spike.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Designed to work in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_Donchian20_Volume_1dTrend_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for Donchian (20) and EMA50 (50)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price breaks above 4h Donchian high + 1d uptrend + volume spike
            if close[i] > donch_high and close[i] > ema50 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below 4h Donchian low + 1d downtrend + volume spike
            elif close[i] < donch_low and close[i] < ema50 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below 4h Donchian low or 1d trend turns down
            if close[i] < donch_low or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises above 4h Donchian high or 1d trend turns up
            if close[i] > donch_high or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals