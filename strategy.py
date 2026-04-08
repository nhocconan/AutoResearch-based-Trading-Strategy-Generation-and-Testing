#!/usr/bin/env python3
"""
6h_1d_1w_fractal_breakout_volume_v1
Hypothesis: Use Williams fractals on daily chart to identify key support/resistance levels.
Enter breakouts on 6h timeframe when price breaks above/below daily fractal levels with
weekly trend filter and volume confirmation. Designed to work in both bull and bear markets
by requiring alignment with higher timeframe trend and institutional volume.
Target: 12-30 trades/year per symbol (48-120 total over 4 years) by requiring confluence
of fractal breakout, trend alignment, and volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "6h_1d_1w_fractal_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Williams fractals on daily high/low
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Fractals need 2 extra bars for confirmation (already handled by compute_williams_fractals?)
    # Actually compute_williams_fractals returns the fractal values, we need to align with delay
    # According to rule 2b: Williams fractal needs 2 extra 1d bars after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Weekly trend filter: price > EMA(21) for bullish, < EMA(21) for bearish
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_bullish = close_1w > ema_21_1w
    weekly_bearish = close_1w < ema_21_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume confirmation: volume > 2x average of last 24 periods (4 days worth)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below bullish fractal support or weekly trend turns bearish
            if close[i] < bullish_fractal_aligned[i] or weekly_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above bearish fractal resistance or weekly trend turns bullish
            if close[i] > bearish_fractal_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above bearish fractal resistance with weekly bullish trend and volume
            if (close[i] > bearish_fractal_aligned[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bullish fractal support with weekly bearish trend and volume
            elif (close[i] < bullish_fractal_aligned[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals