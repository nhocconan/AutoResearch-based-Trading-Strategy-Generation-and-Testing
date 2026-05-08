#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d ADX trend filter
# Long when price breaks above Donchian upper band (20-period high) with volume > 2x 20-day average and ADX > 25
# Short when price breaks below Donchian lower band (20-period low) with volume > 2x 20-day average and ADX > 25
# Exit when price crosses the Donchian midline (10-period average of high/low) or volume drops below average
# Uses daily timeframe for volume and trend confirmation to reduce noise and focus on institutional participation
# Targets 80-150 total trades over 4 years (20-38/year) for optimal risk/reward

name = "4h_Donchian20_1dVolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2x 20-day average
    vol_20ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_20ma * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 1d ADX (14-period) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = WilderSmoothing(tr, 14)
    di_plus_1d = WilderSmoothing(dm_plus, 14)
    di_minus_1d = WilderSmoothing(dm_minus, 14)
    
    # Avoid division by zero
    dx_1d = np.where((di_plus_1d + di_minus_1d) != 0,
                     np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d) * 100,
                     0)
    adx_1d = WilderSmoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        vol_spike_val = vol_spike_aligned[i]
        adx_val = adx_1d_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        donch_mid = donchian_mid[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, ADX > 25
            if close_val > donch_high and vol_spike_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, ADX > 25
            elif close_val < donch_low and vol_spike_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian mid or ADX weakens
            if close_val < donch_mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian mid or ADX weakens
            if close_val > donch_mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals