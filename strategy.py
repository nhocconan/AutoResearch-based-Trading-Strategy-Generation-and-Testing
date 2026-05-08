#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator combination with volume confirmation
# Uses 1d for Williams Alligator (trend) and volume filters, 6h for ADX (momentum)
# Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) for trend direction
# ADX > 25 indicates strong trend for entry, with Alligator alignment for direction
# Volume confirmation reduces false signals
# Works in bull/bear via trend-following logic with volume filter
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ADX_WilliamsAlligator_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (all use median price)
    # Median price = (high + low) / 2
    median_price = (df_daily['high'].values + df_daily['low'].values) / 2
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    jaw_raw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = jaw_raw[7]  # fill shifted values with last valid
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth_raw = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = teeth_raw[4]  # fill shifted values with last valid
    
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips_raw = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = lips_raw[2]  # fill shifted values with last valid
    
    # Calculate daily volume average for volume confirmation
    daily_volume = df_daily['volume'].values
    vol_ma_13 = pd.Series(daily_volume).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_daily, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_daily, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_daily, lips)
    vol_ma_13_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_13)
    
    # Calculate 6h ADX (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
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
            np.isnan(vol_ma_13_aligned[i]) or np.isnan(adx[i])):
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
        
        # Volume filter: current daily volume > 1.5x 13-day EMA
        # Find the most recent completed daily bar
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ma_13_aligned[i]
        
        # ADX filter: > 25 indicates strong trend
        adx_filter = adx[i] > 25
        
        if position == 0:
            # Look for entry with Alligator alignment, ADX strength, and volume
            if alligator_uptrend and adx_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            elif alligator_downtrend and adx_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or ADX weakens
            if not alligator_uptrend or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or ADX weakens
            if not alligator_downtrend or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals