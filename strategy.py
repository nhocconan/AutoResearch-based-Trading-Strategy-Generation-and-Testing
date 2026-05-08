#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day VWAP as trend filter, 12-hour Donchian(20) breakout, and volume confirmation.
# Long when price > 1d VWAP (bullish trend), price breaks above 12h Donchian upper band, volume > 2.0x average.
# Short when price < 1d VWAP (bearish trend), price breaks below 12h Donchian lower band, volume > 2.0x average.
# Exit on trend reversal, Donchian break in opposite direction, or max 20 bars held.
# Uses position size 0.28 to balance return and drawdown. Target: 80-120 total trades over 4 years (20-30/year).
# Designed to capture trends in both bull and bear markets by using 1d VWAP filter, with volume to confirm breakout strength.

name = "12h_1dVWAP_12hDonchian_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 12h data for Donchian bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 1-day VWAP (volume-weighted average price)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (np.cumsum(typical_price_1d * volume_1d) / np.cumsum(volume_1d))
    vwap_1d = np.where(np.cumsum(volume_1d) == 0, 0, vwap_1d)  # avoid division by zero
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 12-hour Donchian(20) bands
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > 1d VWAP (bullish trend), price breaks above 12h Donchian upper band, volume spike
            if (close[i] > vwap_1d_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.28
                position = 1
                entry_bar = i
            # Short: price < 1d VWAP (bearish trend), price breaks below 12h Donchian lower band, volume spike
            elif (close[i] < vwap_1d_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.28
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal, price breaks below Donchian lower band, or max 20 bars held
            if (close[i] < vwap_1d_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Short exit: trend reversal, price breaks above Donchian upper band, or max 20 bars held
            if (close[i] > vwap_1d_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals