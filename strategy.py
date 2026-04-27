#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal breakout with 1-day volume confirmation and 1-day trend filter.
Goes long when bullish fractal forms above 1-day EMA with volume > 1.5x average and price breaks recent high.
Goes short when bearish fractal forms below 1-day EMA with volume > 1.5x average and price breaks recent low.
Designed to capture momentum breaks in both bull and bear markets using fractal structure for entry timing.
Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for fractals, volume filter, and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra bars for confirmation (center + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(25) for trend filter
    close_1d = df_1d['close'].values
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need fractals, volume MA, and daily EMA
    start_idx = max(2, 20, 25)  # fractals need 2 bars, plus MA/EMA lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_25_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_25_1d_aligned[i]
        
        # Current fractal levels (price at which fractal formed)
        bullish_level = bullish_fractal_aligned[i]
        bearish_level = bearish_fractal_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: bullish fractal formed, price above it, volume surge, price above daily EMA
            if (not np.isnan(bullish_level) and 
                price_now > bullish_level and 
                vol_filter and 
                price_now > trend_1d):
                signals[i] = size
                position = 1
            # Short: bearish fractal formed, price below it, volume surge, price below daily EMA
            elif (not np.isnan(bearish_level) and 
                  price_now < bearish_level and 
                  vol_filter and 
                  price_now < trend_1d):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price drops below bullish fractal level or daily trend turns down
            if price_now < bullish_level or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above bearish fractal level or daily trend turns up
            if price_now > bearish_level or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dVolume_1dTrend"
timeframe = "6h"
leverage = 1.0