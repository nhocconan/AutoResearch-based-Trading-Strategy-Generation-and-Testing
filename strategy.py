#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h volume confirmation and 1d EMA50 trend filter.
# Uses 6h Donchian channels (20-period) for breakout signals.
# Confirms with 12h volume > 1.5x 20-period average and 1d EMA50 trend direction.
# Designed for low turnover (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets (breakout momentum) and bear markets (trend-following via EMA filter).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume MA (20-period)
    volume_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma20_12h)
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for Donchian, volume MA, and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(volume_ma20_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period average
        volume_filter = volume_12h[i // 2] > (1.5 * volume_ma20_12h_aligned[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        # Price relative to Donchian channels
        price_above_upper = close[i] > high_max_20[i]
        price_below_lower = close[i] < low_min_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume and above 1d EMA50
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume and below 1d EMA50
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower band OR below 1d EMA50
            if (close[i] < low_min_20[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper band OR above 1d EMA50
            if (close[i] > high_max_20[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hVolume_1dEMA50"
timeframe = "6h"
leverage = 1.0