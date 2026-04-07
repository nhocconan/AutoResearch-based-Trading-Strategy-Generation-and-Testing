#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume and ATR Filter
# Hypothesis: Price breaking above/below daily Donchian(20) channels with volume confirmation
# and ATR-based trend filter works in both bull and bear markets by capturing
# sustained momentum while avoiding false breakouts in choppy conditions.
# Target: 12-30 trades/year (48-120 over 4 years).

name = "12h_donchian20_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate upper and lower bands
    high_series = pd.Series(daily_high)
    low_series = pd.Series(daily_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # ATR filter: ATR(14) > 0.5 * ATR(50) to avoid choppy markets
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift(1))
    tr3 = np.abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_50 = tr.ewm(span=50, min_periods=50, adjust=False).mean().values
    atr_filter = atr_14 > (0.5 * atr_50)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend/volume filter fails
            if (low[i] <= donchian_low_aligned[i]) or \
               (not atr_filter[i]) or (not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend/volume filter fails
            if (high[i] >= donchian_high_aligned[i]) or \
               (not atr_filter[i]) or (not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume and trend
            if (high[i] > donchian_high_aligned[i]) and \
               atr_filter[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume and trend
            elif (low[i] < donchian_low_aligned[i]) and \
                 atr_filter[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals