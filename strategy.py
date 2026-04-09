#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA21) and volume confirmation
# Uses Donchian channels from 6h data: breakout above upper band = long, below lower band = short
# 12h EMA21 filter ensures trades align with higher timeframe trend (more stable than 1d)
# Volume confirmation reduces false breakouts
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: EMA21 adapts to trend, Donchian provides robust structure

name = "6h_12h_donchian_ema_volume_v1"
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
    
    # Load 12h data ONCE before loop for Donchian channels and EMA21
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian bands to 6h timeframe
    upper_20_6h = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_6h = align_htf_to_ltf(prices, df_12h, lower_20)
    
    # Calculate 12h EMA21 trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 12h EMA21 to 6h timeframe
    ema_21_6h = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
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
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or
            np.isnan(ema_21_6h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR price < EMA21 (trend turns bearish)
            if close[i] < lower_20_6h[i] or close[i] < ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR price > EMA21 (trend turns bullish)
            if close[i] > upper_20_6h[i] or close[i] > ema_21_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > EMA21 (bullish trend)
                if close[i] > upper_20_6h[i] and close[i] > ema_21_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < EMA21 (bearish trend)
                elif close[i] < lower_20_6h[i] and close[i] < ema_21_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals