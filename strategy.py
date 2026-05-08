#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX + 4h Donchian breakout with volume confirmation and session filter
# Long when 4h price breaks above Donchian(20) upper band + 1h ADX > 25 + volume spike + session (08-20 UTC)
# Short when 4h price breaks below Donchian(20) lower band + 1h ADX > 25 + volume spike + session
# Uses 4h for trend structure (Donchian channels), 1h for entry timing and ADX filter
# Volume spike confirms institutional participation
# Session filter reduces noise during low-liquidity hours
# Targets 60-150 total trades over 4 years (15-37/year) to avoid fee drag

name = "1h_ADX_DonchianBreakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data once for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 1h ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    plus_di = 100 * wilder_smooth(plus_dm, 14) / wilder_smooth(tr, 14)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / wilder_smooth(tr, 14)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilder_smooth(dx, 14)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 4h price breaks above Donchian high + ADX > 25 + volume spike + session
            if (high[i] > donchian_high_aligned[i] and 
                adx[i] > 25 and 
                volume_spike[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: 4h price breaks below Donchian low + ADX > 25 + volume spike + session
            elif (low[i] < donchian_low_aligned[i] and 
                  adx[i] > 25 and 
                  volume_spike[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR ADX drops below 20
            if low[i] < donchian_low_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Donchian high OR ADX drops below 20
            if high[i] > donchian_high_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals