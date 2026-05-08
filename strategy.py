#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX Trend Filter + Volume Spike
# Uses daily ADX(14) > 25 to identify trending markets, Williams %R(14) for mean reversion entries
# in the direction of the trend, and volume spike (>2x average) for confirmation.
# Works in both bull and bear markets by following daily trend while avoiding range-bound conditions.
# Target: 20-40 trades/year.

name = "6h_WilliamsR_1dADX_Trend_VolumeSpike"
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
    
    # Get daily data for ADX trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 28:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_daily[1:] - high_daily[:-1]) > (low_daily[:-1] - low_daily[1:]), 
                       np.maximum(high_daily[1:] - high_daily[:-1], 0), 0)
    dm_minus = np.where((low_daily[:-1] - low_daily[1:]) > (high_daily[1:] - high_daily[:-1]), 
                        np.maximum(low_daily[:-1] - low_daily[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmoothing(tr, 14)
    dm_plus14 = WilderSmoothing(dm_plus, 14)
    dm_minus14 = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(tr14, np.nan)
    di_minus = np.full_like(tr14, np.nan)
    mask = ~np.isnan(tr14) & (tr14 != 0)
    di_plus[mask] = 100 * dm_plus14[mask] / tr14[mask]
    di_minus[mask] = 100 * dm_minus14[mask] / tr14[mask]
    
    # DX and ADX
    dx = np.full_like(tr14, np.nan)
    mask_dx = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[mask_dx] = 100 * np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / (di_plus[mask_dx] + di_minus[mask_dx])
    
    adx = WilderSmoothing(dx, 14)
    
    # Calculate daily Williams %R(14)
    highest_high_14 = np.full_like(high_daily, np.nan)
    lowest_low_14 = np.full_like(low_daily, np.nan)
    for i in range(13, len(high_daily)):
        highest_high_14[i] = np.max(high_daily[i-13:i+1])
        lowest_low_14[i] = np.min(low_daily[i-13:i+1])
    
    williams_r = np.full_like(close_daily, np.nan)
    for i in range(13, len(close_daily)):
        if highest_high_14[i] != lowest_low_14[i]:
            williams_r[i] = -100 * (highest_high_14[i] - close_daily[i]) / (highest_high_14[i] - lowest_low_14[i])
        else:
            williams_r[i] = -50  # neutral
    
    # Calculate daily volume average
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full_like(vol_daily, np.nan)
    for i in range(19, len(vol_daily)):
        vol_avg_20_daily[i] = np.mean(vol_daily[i-19:i+1])
    
    # Align daily indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_daily, williams_r)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(27, 19)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R extremes in trending market with volume spike
            trending_market = adx_aligned[i] > 25
            
            # Long when Williams %R oversold (< -80) in uptrend
            long_condition = (
                williams_r_aligned[i] < -80 and   # oversold condition
                trending_market and               # trending market (ADX > 25)
                vol_spike                         # volume spike for confirmation
            )
            
            # Short when Williams %R overbought (> -20) in downtrend
            short_condition = (
                williams_r_aligned[i] > -20 and   # overbought condition
                trending_market and               # trending market (ADX > 25)
                vol_spike                         # volume spike for confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend weakens
            if williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend weakens
            if williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals