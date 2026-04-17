#!/usr/bin/env python3
"""
12h 1W Donchian Breakout + Volume Spike + 1D ADX Trend Filter
Long: Price breaks above 1W Donchian upper + volume > 2x 12h volume SMA(20) + 1D ADX > 25
Short: Price breaks below 1W Donchian lower + volume > 2x 12h volume SMA(20) + 1D ADX > 25
Exit: Opposite breakout or ADX < 20
Uses weekly structure for major trend, volume for conviction, daily ADX for trend strength.
Designed to work in trending markets (both bull and bear) while avoiding chop.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1W data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for weekly close)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # Get 1D data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = WilderSmoothing(tr, 14)
    plus_di_14 = 100 * WilderSmoothing(plus_dm, 14) / atr_14
    minus_di_14 = 100 * WilderSmoothing(minus_dm, 14) / atr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = WilderSmoothing(dx_14, 14)
    
    # Handle division by zero and NaN
    adx_14 = np.where((plus_di_14 + minus_di_14) == 0, 0, adx_14)
    adx_14 = np.where(np.isnan(adx_14), 0, adx_14)
    
    # Align ADX to 12h timeframe (wait for daily close)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 12h volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need sufficient data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        donch_high = donch_high_20_aligned[i]
        donch_low = donch_low_20_aligned[i]
        adx_val = adx_14_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + volume spike + strong trend (ADX > 25)
            if price > donch_high and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + volume spike + strong trend (ADX > 25)
            elif price < donch_low and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below weekly Donchian low OR trend weakens (ADX < 20)
            if price < donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above weekly Donchian high OR trend weakens (ADX < 20)
            if price > donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1WDonchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0