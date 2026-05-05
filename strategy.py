#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume spike and 1d ADX trend filter
# Long when price breaks above 6h Donchian upper band AND 12h volume > 2x 20-period average AND 1d ADX > 25
# Short when price breaks below 6h Donchian lower band AND 12h volume > 2x 20-period average AND 1d ADX > 25
# Exit when price crosses 6h Donchian midpoint (mean reversion within the channel)
# Uses 6h primary timeframe with 12h HTF for volume confirmation and 1d HTF for trend strength
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Donchian channels provide clear breakout levels; volume confirms institutional participation; ADX filters for trending markets only

name = "6h_Donchian20_Breakout_12hVolumeSpike_1dADX25_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period)
    if len(high) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (highest_high + lowest_low) / 2
    else:
        highest_high = high
        lowest_low = low
        donchian_mid = close
    
    # Calculate 12h volume confirmation: volume > 2x 20-period average
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        volume_spike_12h = vol_12h > (2.0 * vol_ma_20_12h)
        volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    else:
        volume_spike_12h_aligned = np.zeros(n, dtype=bool)
    
    # Calculate 1d ADX (14-period) for trend strength filter
    if len(df_1d) >= 30:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values (Wilder's smoothing)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            first_val = np.nansum(data[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr_14 = wilders_smoothing(tr, 14)
        dm_plus_14 = wilders_smoothing(dm_plus, 14)
        dm_minus_14 = wilders_smoothing(dm_minus, 14)
        
        # DI+ and DI-
        di_plus = np.where(atr_14 != 0, (dm_plus_14 / atr_14) * 100, 0)
        di_minus = np.where(atr_14 != 0, (dm_minus_14 / atr_14) * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = wilders_smoothing(dx, 14)
        
        # ADX > 25 indicates strong trend
        adx_filter = adx > 25
        adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    else:
        adx_filter_aligned = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND strong trend (ADX > 25)
            if (close[i] > highest_high[i] and 
                volume_spike_12h_aligned[i] and 
                adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND strong trend (ADX > 25)
            elif (close[i] < lowest_low[i] and 
                  volume_spike_12h_aligned[i] and 
                  adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion within channel)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion within channel)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals