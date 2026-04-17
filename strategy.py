#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 20-day Donchian breakout with 1-week EMA34 trend filter and volume confirmation.
# Enters long when price breaks above 20-day high with volume and above weekly EMA34.
# Enters short when price breaks below 20-day low with volume and below weekly EMA34.
# Designed to capture strong trends while avoiding whipsaws in ranging markets.
# Uses weekly EMA34 for trend filter to avoid false breakouts in weak trends.
# Volume filter ensures breakouts have institutional participation.
# Target: 15-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels (using previous day's data to avoid look-ahead)
    # Highest high of previous 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of previous 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1-week EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Donchian channels to daily timeframe (already aligned since we're using 1d data)
    donchian_high_1d = donchian_high
    donchian_low_1d = donchian_low
    
    # Align 1w EMA34 to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average (moderate to balance signal quality and frequency)
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1w_aligned[i]
        price_below_ema = close[i] < ema34_1w_aligned[i]
        
        # Price relative to Donchian channels
        price_above_high = close[i] > donchian_high_1d[i]
        price_below_low = close[i] < donchian_low_1d[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume and above weekly EMA34
            if (price_above_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume and below weekly EMA34
            elif (price_below_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below 20-day low OR below weekly EMA34
            if (close[i] < donchian_low_1d[i]) or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above 20-day high OR above weekly EMA34
            if (close[i] > donchian_high_1d[i]) or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0