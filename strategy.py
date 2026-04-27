#!/usr/bin/env python3
"""
12h Williams Fractal Breakout with 1d Trend Filter and Volume Spike.
Long when: 1) Bullish fractal breakout above recent high, 2) Price > 1d EMA50 (bullish trend), 3) Volume > 2x 20-period average.
Short when: 1) Bearish fractal breakdown below recent low, 2) Price < 1d EMA50 (bearish trend), 3) Volume > 2x 20-period average.
Exit when price crosses 1d EMA50 (trend reversal).
Designed for 12h timeframe: targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for EMA50 trend filter and Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Fractals (need 2-bar confirmation after center bar)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d EMA (50 periods), fractals, volume MA (20 periods)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        bullish_fractal = bullish_fractal_aligned[i]
        bearish_fractal = bearish_fractal_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: bullish fractal breakout + bullish trend + volume spike
            if bullish_fractal and price > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: bearish fractal breakdown + bearish trend + volume spike
            elif bearish_fractal and price < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend turns bearish (price < EMA50)
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend turns bullish (price > EMA50)
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Fractal_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0