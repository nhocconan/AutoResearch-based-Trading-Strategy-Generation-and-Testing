#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume spike.
# Uses price channels (Donchian) for entries, 1d EMA for trend filter, volume for confirmation.
# Designed to work in bull (breakouts with trend) and bear (breakouts against trend).
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily Donchian and EMA to 12h
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 50-period average (moderate to reduce trades)
    volume_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily EMA50 and volume MA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_12h[i]) or 
            np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to reduce trades)
        volume_filter = volume[i] > (1.5 * volume_ma50[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        # Price relative to daily Donchian levels
        price_above_upper = close[i] > donchian_upper_12h[i]
        price_below_lower = close[i] < donchian_lower_12h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and above daily EMA50
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and below daily EMA50
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower OR below daily EMA50
            if (close[i] < donchian_lower_12h[i]) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper OR above daily EMA50
            if (close[i] > donchian_upper_12h[i]) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_Volume"
timeframe = "12h"
leverage = 1.0