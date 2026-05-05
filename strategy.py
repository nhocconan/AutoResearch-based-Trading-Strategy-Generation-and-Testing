#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# Long when price breaks above Donchian upper AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Short when price breaks below Donchian lower AND volume > 1.5x 20-period average AND 1w ADX > 25 (trending)
# Exit when price crosses back to Donchian midpoint OR 1w ADX < 20 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian provides structural breakouts, volume spike confirms conviction, 1w ADX filters for trending markets.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Donchian20_VolumeSpike_1wADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume calculation (if needed) and Donchian context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for Donchian and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1w data (using previous 20 periods to avoid look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper: highest high over past 20 periods
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower: lowest low over past 20 periods
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of upper and lower
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate ADX on 1w data (trend strength filter)
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], high_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], low_1w[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # first TR is just high-low
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Trend filter: ADX > 25 indicates trending market
    trending = adx > 25
    ranging = adx < 20  # exit condition
    
    # Align ADX-based filters to 12h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1w, trending.astype(float))
    ranging_aligned = align_htf_to_ltf(prices, df_1w, ranging.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (using 12h data)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(trending_aligned[i]) or 
            np.isnan(ranging_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND trending
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND trending
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian low OR market becomes ranging
            if (close[i] < donchian_low_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian high OR market becomes ranging
            if (close[i] > donchian_high_aligned[i] or 
                ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals