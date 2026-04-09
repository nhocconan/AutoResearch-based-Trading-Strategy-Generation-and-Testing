#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d HMA50 trend filter and volume confirmation
# Uses Donchian channels from 6h data: breakout above upper band = long, below lower band = short
# 1d HMA50 filter ensures trades align with higher timeframe trend (more stable than 12h)
# Volume confirmation reduces false breakouts
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: HMA50 adapts to trend, Donchian provides robust structure

name = "6h_1d_donchian_hma_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian channels and HMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 6h data
    high_6h = high
    low_6h = low
    close_6h = close
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA50 trend filter
    df_1d_close = df_1d['close'].values
    half_n = int(50/2 + 0.5)
    wma_half = pd.Series(df_1d_close).rolling(window=half_n, min_periods=half_n).mean()
    wma_full = pd.Series(df_1d_close).rolling(window=50, min_periods=50).mean()
    hma_50_1d = (2 * wma_half - wma_full).values
    
    # Align 1d HMA50 to 6h timeframe
    hma_50_6h = align_htf_to_ltf(prices, df_1d, hma_50_1d)
    
    # Calculate 20-period average volume for volume confirmation (6h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(hma_50_6h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend turns bearish
            if close[i] < lower_20[i] or close[i] < hma_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend turns bullish
            if close[i] > upper_20[i] or close[i] > hma_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > 1d HMA50 (bullish trend)
                if close[i] > upper_20[i] and close[i] > hma_50_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < 1d HMA50 (bearish trend)
                elif close[i] < lower_20[i] and close[i] < hma_50_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals