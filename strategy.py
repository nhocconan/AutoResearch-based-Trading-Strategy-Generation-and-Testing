#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(30) breakout with 1d EMA100 trend filter and volume spike confirmation.
# Uses longer lookback (30) and slower EMA (100) to reduce trade frequency.
# Volume spike > 1.5x 20-period average confirms breakout strength.
# Designed for low turnover in both bull and bear markets by requiring strong trends.
# Target: 15-25 trades/year to stay within optimal range for 12h timeframe.

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
    
    # Calculate 1d Donchian channels (30-period for fewer signals)
    high_max_30 = pd.Series(high_1d).rolling(window=30, min_periods=30).max().values
    low_min_30 = pd.Series(low_1d).rolling(window=30, min_periods=30).min().values
    
    # Calculate 1d EMA100 for stronger trend filter
    close_1d_series = pd.Series(close_1d)
    ema100_1d = close_1d_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d Donchian and EMA to 12h
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, high_max_30)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, low_min_30)
    ema100_12h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume filter: spike > 1.5x 20-period average (moderate to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 30-period Donchian (1d) + EMA100 (1d) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema100_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to reduce trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA100
        price_above_ema = close[i] > ema100_12h[i]
        price_below_ema = close[i] < ema100_12h[i]
        
        # Price relative to 1d Donchian channels
        price_above_high = close[i] > donchian_high_12h[i]
        price_below_low = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high with volume and above 1d EMA100
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian low with volume and below 1d EMA100
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 1d Donchian low OR below 1d EMA100
            if (close[i] < donchian_low_12h[i]) or (close[i] < ema100_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 1d Donchian high OR above 1d EMA100
            if (close[i] > donchian_high_12h[i]) or (close[i] > ema100_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian30_1dEMA100_Volume"
timeframe = "12h"
leverage = 1.0