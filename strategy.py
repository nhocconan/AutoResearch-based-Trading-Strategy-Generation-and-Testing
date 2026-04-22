#!/usr/bin/env python3
"""
Hypothesis: 1-day ADX (trend strength) + 1-week Donchian breakout (trend direction) with volume confirmation.
Long when weekly price breaks above Donchian high with daily ADX > 25 and volume spike.
Short when weekly price breaks below Donchian low with daily ADX > 25 and volume spike.
Exit when price crosses back below/above Donchian mid-line or ADX weakens (<20).
Uses weekly trend structure with daily momentum filter to avoid whipsaws in ranging markets.
Designed for low trade frequency by requiring multiple confirmations and weekly breakouts.
Works in both bull and bear markets by following the dominant weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high and low (20-period)
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high_20 + donch_low_20) / 2.0
    
    # Align to daily timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # Load daily data for ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = WilderSmoothing(tr, 14)
    plus_dm14 = WilderSmoothing(plus_dm, 14)
    minus_dm14 = WilderSmoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, 14)
    
    # Align to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for ADX
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with daily ADX > 25 and volume spike
            if (close[i] > donch_high_20_aligned[i] and 
                adx_aligned[i] > 25 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with daily ADX > 25 and volume spike
            elif (close[i] < donch_low_20_aligned[i] and 
                  adx_aligned[i] > 25 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below weekly Donchian mid-line OR ADX weakens (<20)
                if (close[i] < donch_mid_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above weekly Donchian mid-line OR ADX weakens (<20)
                if (close[i] > donch_mid_aligned[i] or adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_ADX_WeeklyDonchianBreakout_Volume"
timeframe = "1d"
leverage = 1.0