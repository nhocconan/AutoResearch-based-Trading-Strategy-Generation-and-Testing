#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Donchian channels provide clear breakout levels. 12h HMA confirms medium-term trend direction.
# Volume spike validates breakout strength. Works in bull via upside breakouts, in bear via downside breakouts.
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h HMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) channels from 12h data
    donchian_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 21)  # warmup for Donchian and HMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(hma_12h_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_hma = hma_12h_aligned[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above Donchian high AND above 12h HMA (uptrend)
                if curr_close > curr_donchian_high and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below Donchian low AND below 12h HMA (downtrend)
                elif curr_close < curr_donchian_low and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low (mean reversion)
            if curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high (mean reversion)
            if curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(arr).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    # WMA of full period
    wma_full = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw HMA with sqrt period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    return hma