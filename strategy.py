#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 12h volume spike and 1d ADX trend filter
# Long when Williams %R < -80 (oversold) AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Short when Williams %R > -20 (overbought) AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending)
# Exit when Williams %R returns to -50 level OR ADX < 20 (trend weakens)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Williams %R identifies exhaustion points, volume spike confirms participation,
# ADX ensures we only trade in trending markets to avoid chop whipsaws.
# Works in bull markets via buying oversold dips in uptrends and bear markets via selling overbought rallies in downtrends.

name = "6h_WilliamsR_EXTREME_12hVolumeSpike_1dADX_Trend"
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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h volume MA20 for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = vol_12h > (1.5 * vol_ma_20_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Trend filters: ADX > 25 for trending, ADX < 20 for ranging
    adx_trending = adx > 25
    adx_ranging = adx < 20
    
    # Align 1d ADX to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    # Williams %R on 6h data (primary timeframe)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(volume_spike_12h_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND volume spike AND ADX trending
            if (williams_r[i] < -80 and 
                volume_spike_12h_aligned[i] > 0.5 and 
                adx_trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND volume spike AND ADX trending
            elif (williams_r[i] > -20 and 
                  volume_spike_12h_aligned[i] > 0.5 and 
                  adx_trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR ADX becomes ranging (<20)
            if (williams_r[i] > -50 or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR ADX becomes ranging (<20)
            if (williams_r[i] < -50 or 
                adx_ranging_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals