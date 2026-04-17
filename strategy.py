#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Uses 4h Donchian(20) upper/lower bands for breakout signals.
# Enters long when price breaks above Donchian upper band with volume and above 12h EMA50.
# Enters short when price breaks below Donchian lower band with volume and below 12h EMA50.
# Exits on Donchian midpoint or EMA crossover.
# Designed to capture momentum breakouts with low turnover (target: 15-30 trades/year).
# Works in bull markets (breakout momentum) and bear markets (mean reversion via reversion to mean).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema50_12h_aligned[i]
        price_below_ema = close[i] < ema50_12h_aligned[i]
        
        # Price relative to Donchian bands
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume and above 12h EMA50
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume and below 12h EMA50
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian middle OR below 12h EMA50
            if (close[i] < donchian_middle[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian middle OR above 12h EMA50
            if (close[i] > donchian_middle[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0