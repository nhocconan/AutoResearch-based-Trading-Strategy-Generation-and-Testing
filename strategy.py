#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses price channels (Donchian) for breakouts, weekly EMA for trend filter, volume spike for confirmation.
# Designed to work in bull (upward breakouts with trend) and bear (downward breakouts with trend).
# Target: 12-37 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate weekly EMA50 for trend filter
    close_weekly_series = pd.Series(close_weekly)
    ema50_weekly = close_weekly_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA and 12h Donchian to 12h
    ema50_12h = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, high_max_20)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # Volume filter: current volume > 1.5 * 20-period average (reduces trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need 20-period Donchian (12h) + EMA50 (weekly) + volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (strict to reduce trades)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        # Price relative to 12h Donchian channels
        price_above_high = close[i] > donchian_high_12h[i]
        price_below_low = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume and above weekly EMA50
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian low with volume and below weekly EMA50
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 12h Donchian low OR below weekly EMA50
            if (close[i] < donchian_low_12h[i]) or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 12h Donchian high OR above weekly EMA50
            if (close[i] > donchian_high_12h[i]) or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0