#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams Fractal breakout with 1-week trend filter and volume confirmation.
- Go long when price breaks above a confirmed weekly bearish fractal (resistance) with volume > 1.5x average.
- Go short when price breaks below a confirmed weekly bullish fractal (support) with volume > 1.5x average.
- Weekly fractals require 2 additional bars for confirmation (total 5-bar pattern: low, lower, lowest, lower, low).
- Uses weekly trend direction to filter trades: only trade in direction of weekly EMA(34) trend.
- Designed for low frequency (~15-30 trades/year) to capture institutional breakouts while avoiding false signals.
- Works in bull markets by catching breakouts above weekly resistance.
- Works in bear markets by catching breakdowns below weekly support.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for fractals and trend - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Williams Fractals (requires 5 bars: 2 left, 1 center, 2 right)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    
    # Weekly fractals need 2 additional bars for confirmation after the center bar
    bearish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_confirmed = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(bearish_fractal_confirmed[i]) or np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current levels
        bearish_fractal_level = bearish_fractal_confirmed[i]
        bullish_fractal_level = bullish_fractal_confirmed[i]
        ema_34_1w_level = ema_34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long conditions:
            # 1. Price breaks above weekly bearish fractal (resistance)
            # 2. Weekly trend is up (price > EMA34)
            # 3. Volume confirmation
            if (close[i] > bearish_fractal_level and 
                close[i] > ema_34_1w_level and
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price breaks below weekly bullish fractal (support)
            # 2. Weekly trend is down (price < EMA34)
            # 3. Volume confirmation
            elif (close[i] < bullish_fractal_level and 
                  close[i] < ema_34_1w_level and
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse fractal break of opposite type
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below weekly bullish fractal (support)
                if close[i] < bullish_fractal_level:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above weekly bearish fractal (resistance)
                if close[i] > bearish_fractal_level:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsFractal_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0