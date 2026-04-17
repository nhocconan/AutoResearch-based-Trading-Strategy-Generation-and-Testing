#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h breakout at 12h Donchian(15) with 1d EMA21 trend filter and volume confirmation.
# Uses 12h Donchian breakout levels for entry timing and 1d EMA21 for trend alignment.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 12-30 trades/year to stay within optimal range for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Get 1d data for EMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (15-period)
    high_max_15 = pd.Series(high_12h).rolling(window=15, min_periods=15).max().values
    low_min_15 = pd.Series(low_12h).rolling(window=15, min_periods=15).min().values
    
    # Calculate 1d EMA21 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema21_1d = close_1d_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 12h Donchian and 1d EMA to 6h
    donchian_high_6h = align_htf_to_ltf(prices, df_12h, high_max_15)
    donchian_low_6h = align_htf_to_ltf(prices, df_12h, low_min_15)
    ema21_6h = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need 15-period Donchian (12h) + EMA21 (1d) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_6h[i]) or 
            np.isnan(donchian_low_6h[i]) or 
            np.isnan(ema21_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA21
        price_above_ema = close[i] > ema21_6h[i]
        price_below_ema = close[i] < ema21_6h[i]
        
        # Price relative to 12h Donchian channels
        price_above_high = close[i] > donchian_high_6h[i]
        price_below_low = close[i] < donchian_low_6h[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume and above 1d EMA21
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low with volume and below 1d EMA21
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 12h Donchian low OR below 1d EMA21
            if (close[i] < donchian_low_6h[i]) or (close[i] < ema21_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 12h Donchian high OR above 1d EMA21
            if (close[i] > donchian_high_6h[i]) or (close[i] > ema21_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian15_1dEMA21_Volume"
timeframe = "6h"
leverage = 1.0