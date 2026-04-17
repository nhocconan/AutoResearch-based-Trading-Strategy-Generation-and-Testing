#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA200 trend filter and volume confirmation.
# Uses 1d Donchian upper/lower bands (20-period high/low) and EMA200 for trend bias.
# Enters long when price breaks above 1d Donchian upper band with volume and above EMA200.
# Enters short when price breaks below 1d Donchian lower band with volume and below EMA200.
# Exits when price crosses the 1d Donchian midline (average of upper/lower bands).
# Designed for low turnover (target: 12-37 trades/year) and works in both bull and bear markets
# by following the higher-timeframe trend while using breakouts for entry timing.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and EMA200
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donchian_high = high_1d_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d EMA200 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_12h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema200_12h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or 
            np.isnan(donchian_mid_12h[i]) or 
            np.isnan(ema200_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema200_12h[i]
        price_below_ema = close[i] < ema200_12h[i]
        
        # Price relative to Donchian levels
        price_above_high = close[i] > donchian_high_12h[i]
        price_below_low = close[i] < donchian_low_12h[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and above EMA200
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and below EMA200
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian midline OR below EMA200
            if (close[i] < donchian_mid_12h[i]) or (close[i] < ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian midline OR above EMA200
            if (close[i] > donchian_mid_12h[i]) or (close[i] > ema200_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA200_Volume"
timeframe = "12h"
leverage = 1.0