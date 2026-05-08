#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d Williams Alligator trend alignment + volume confirmation
# Uses 4h Donchian channel (20) for structure, 1d Williams Alligator for trend, and 4h volume spike
# Effective in bull/bear via trend-following with volume filter
# Target: 50-150 total trades over 4 years (12-38/year) to minimize fee drag

name = "4h_Donchian_Alligator_1dTrend_Volume"
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
    
    # Get daily data for Williams Alligator
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (all use median price)
    median_price = (df_daily['high'].values + df_daily['low'].values) / 2
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    jaw_raw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = jaw_raw[7]
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth_raw = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = teeth_raw[4]
    
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips_raw = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = lips_raw[2]
    
    # Align daily indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    
    # Calculate 4h Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 4h volume average for volume spike detection
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator trend: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        alligator_uptrend = lips_val > teeth_val > jaw_val
        alligator_downtrend = lips_val < teeth_val < jaw_val
        
        # Volume filter: current volume > 2x 20-period EMA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout + Alligator alignment + volume
            if close[i] > donchian_high[i-1] and alligator_uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i-1] and alligator_downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Donchian break below lower band or Alligator alignment breaks
            if close[i] < donchian_low[i] or not alligator_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Donchian break above upper band or Alligator alignment breaks
            if close[i] > donchian_high[i] or not alligator_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals