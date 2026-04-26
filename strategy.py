#!/usr/bin/env python3
"""
6h_Williams_Fractal_Breakout_WeeklyTrend_v1
Hypothesis: 6h Williams fractal breakout strategy with weekly trend filter.
- Uses 6h timeframe for moderate trade frequency (target: 50-150 total trades over 4 years)
- Williams fractals identify potential reversal points (5-bar patterns: highest high/lowest low surrounded by lower highs/higher lows)
- Breakouts above/below recent fractal levels with volume confirmation
- Weekly EMA50 filter ensures trades align with higher timeframe trend
- Long when price breaks above recent bullish fractal resistance with volume spike AND weekly uptrend
- Short when price breaks below recent bearish fractal support with volume spike AND weekly downtrend
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the weekly trend and using fractals for structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Load daily data for Williams fractals (need extra delay for confirmation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra daily bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume average for volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, plus fractal alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma20[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above recent bullish fractal resistance with volume spike AND weekly uptrend
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] > bullish_fractal_aligned[i] and 
                volume_spike and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below recent bearish fractal support with volume spike AND weekly downtrend
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  close[i] < bearish_fractal_aligned[i] and 
                  volume_spike and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below recent bullish fractal support OR weekly trend turns down
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] < bullish_fractal_aligned[i]) or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above recent bearish fractal resistance OR weekly trend turns up
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i]) or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Williams_Fractal_Breakout_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0