#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses daily EMA for trend filter and 1d Donchian for breakout levels, aligned to 4h.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 15-30 trades/year to stay within optimal range for 4h timeframe.
# Works in both bull and bear by following trend via EMA34 filter and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Donchian and EMA to 4h
    donchian_high_4h = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_4h = align_htf_to_ltf(prices, df_1d, low_min_20)
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian (1d) + EMA34 (1d) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema34_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_4h[i]
        price_below_ema = close[i] < ema34_4h[i]
        
        # Price relative to 1d Donchian channels
        price_above_high = close[i] > donchian_high_4h[i]
        price_below_low = close[i] < donchian_low_4h[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high with volume and above 1d EMA34
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian low with volume and below 1d EMA34
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 1d Donchian low OR below 1d EMA34
            if (close[i] < donchian_low_4h[i]) or (close[i] < ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 1d Donchian high OR above 1d EMA34
            if (close[i] > donchian_high_4h[i]) or (close[i] > ema34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0