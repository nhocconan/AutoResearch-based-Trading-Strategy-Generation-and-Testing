#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20-period) with 1d EMA200 trend filter and volume confirmation.
# Uses 4h Donchian channels (upper/lower bounds) for breakout direction and 1d EMA200 for long-term trend.
# Enters long when price breaks above 4h Donchian upper with volume and above 1d EMA200.
# Enters short when price breaks below 4h Donchian lower with volume and below 1d EMA200.
# Designed to capture institutional breakouts with low turnover (target: 15-37 trades/year).
# Works in bull markets (breakout momentum) and bear markets (trend following via EMA200 filter).
# Uses 4h/1d for signal direction, 1h only for entry timing.

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channel (20-period)
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h and 1d indicators to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    ema200_1h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_1h[i]) or 
            np.isnan(donchian_lower_1h[i]) or 
            np.isnan(ema200_1h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance signal quality and frequency)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema200_1h[i]
        price_below_ema = close[i] < ema200_1h[i]
        
        # Price relative to Donchian levels
        price_above_upper = close[i] > donchian_upper_1h[i]
        price_below_lower = close[i] < donchian_lower_1h[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper with volume and above 1d EMA200
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian lower with volume and below 1d EMA200
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian lower OR below 1d EMA200
            if (close[i] < donchian_lower_1h[i]) or (close[i] < ema200_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian upper OR above 1d EMA200
            if (close[i] > donchian_upper_1h[i]) or (close[i] > ema200_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_1dEMA200_Volume"
timeframe = "1h"
leverage = 1.0