#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h breakout at 4h Donchian(20) with 1d EMA50 trend filter and volume confirmation.
# Uses 4h Donchian breakout levels for entry and 1d EMA50 for trend alignment.
# Volume spike confirms breakout strength. Designed to capture strong trends with low turnover.
# Target: 20-50 trades/year to stay within optimal range for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for EMA
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Donchian and 1d EMA to 4h (no alignment needed as they are already on 4h)
    # But we still use align_htf_to_ltf to ensure proper delay handling
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, high_max_20)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, low_min_20)
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.8 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need 20-period Donchian (4h) + EMA50 (1d) + volume MA30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_4h[i]) or 
            np.isnan(donchian_low_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average (strict to reduce trades)
        volume_filter = volume[i] > (1.8 * volume_ma30[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        # Price relative to 4h Donchian channels
        price_above_high = close[i] > donchian_high_4h[i]
        price_below_low = close[i] < donchian_low_4h[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume and above 1d EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 4h Donchian low with volume and below 1d EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian low OR below 1d EMA50
            if (close[i] < donchian_low_4h[i]) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian high OR above 1d EMA50
            if (close[i] > donchian_high_4h[i]) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0