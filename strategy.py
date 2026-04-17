#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter (EMA50) and volume confirmation.
# Enters long when price breaks above Donchian upper band (20-period high) with volume and above 1d EMA50.
# Enters short when price breaks below Donchian lower band (20-period low) with volume and below 1d EMA50.
# Exit when price reverses to Donchian midpoint or crosses 1d EMA50.
# Designed for low turnover (target: 20-50 trades/year) with strong trend capture.
# Works in bull markets (breakout momentum) and bear markets (trend following via EMA filter).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d EMA50 to 4h timeframe
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need sufficient data for Donchian (20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        # Price relative to Donchian channels
        price_above_upper = close[i] > donchian_high[i]
        price_below_lower = close[i] < donchian_low[i]
        
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
            # Exit long: Price crosses below Donchian midpoint OR below 1d EMA50
            if (close[i] < donchian_mid[i]) or (close[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian midpoint OR above 1d EMA50
            if (close[i] > donchian_mid[i]) or (close[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0