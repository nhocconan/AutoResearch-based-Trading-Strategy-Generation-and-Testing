# 141957: 4h Donchian Breakout + Volume Confirmation + ATR Filter  
# Hypothesis: Breakouts of 4h Donchian channels (20-period) filtered by volume spikes (>2x 20-period average) and ATR-based trend strength.  
# Uses ATR to confirm directional momentum and avoid false breakouts in choppy markets.  
# Designed for 4h timeframe with tight entry conditions to limit trades (target: 20-50/year) and reduce fee drag.  
# Works in both bull and bear markets by capturing breakouts in the direction of momentum.  

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DonchianBreakout_VolumeATR_Filter"
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
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR(14) for trend strength and volatility filter
    def calculate_atr(high_arr, low_arr, close_arr, period):
        tr = np.zeros(len(high_arr))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(high_arr)):
            tr[i] = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
        atr = np.full(len(high_arr), np.nan)
        if len(high_arr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(high_arr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate 4h volume average for volume spike filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Get daily data for optional trend filter (can be used if needed)
    df_daily = get_htf_data(prices, '1d')
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 2x 20-period average
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        # ATR filter: only trade when ATR is above its 20-period average (avoid choppy markets)
        if i >= 20:
            atr_avg_20 = np.mean(atr[i-20:i])
            atr_filter = atr[i] > atr_avg_20
        else:
            atr_filter = False
        
        if position == 0:
            # Look for breakout entry with volume and ATR confirmation
            if close[i] > donchian_high[i] and vol_spike and atr_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and vol_spike and atr_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or loses volume/ATR confirmation
            if close[i] < donchian_low[i] or not vol_spike or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or loses volume/ATR confirmation
            if close[i] > donchian_high[i] or not vol_spike or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals