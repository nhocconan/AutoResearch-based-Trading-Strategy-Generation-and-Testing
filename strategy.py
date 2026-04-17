#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Enters long when price breaks above Donchian upper band with volume and above 1d EMA200.
# Enters short when price breaks below Donchian lower band with volume and below 1d EMA200.
# Exits when price crosses back below/above Donchian mid-band or reverses trend.
# Designed to capture strong breakouts with low turnover (target: 19-50 trades/year).
# Works in bull markets (breakout momentum) and bear markets (trend filter avoids false breakouts).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume filter: current volume > 1.8 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_4h[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average (strict to reduce trades)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema200_4h[i]
        price_below_ema = close[i] < ema200_4h[i]
        
        # Price relative to Donchian channels
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        price_above_middle = close[i] > donchian_middle[i]
        price_below_middle = close[i] < donchian_middle[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume and above 1d EMA200
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume and below 1d EMA200
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian middle OR below 1d EMA200
            if (price_below_middle) or (not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian middle OR above 1d EMA200
            if (price_above_middle) or (not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0