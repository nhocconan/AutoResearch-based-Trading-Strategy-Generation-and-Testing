#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 12h bearish Williams fractal AND price > 1d EMA34 AND volume > 1.5x 20-period average
- Short when price breaks below 12h bullish Williams fractal AND price < 1d EMA34 AND volume > 1.5x 20-period average
- Exit when price crosses the 12h midpoint between the last bullish and bearish fractal levels
- Williams fractals require 2-bar confirmation (additional_delay_bars=2) to avoid look-ahead
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for Williams fractals (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Williams fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_12h['high'].values,
        df_12h['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_12h, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 12h midpoint between last bullish and bearish fractal
    # Use forward fill to propagate last valid fractal level
    bullish_series = pd.Series(bullish_fractal_aligned)
    bearish_series = pd.Series(bearish_fractal_aligned)
    bullish_ffilled = bullish_series.ffill().values
    bearish_ffilled = bearish_series.ffill().values
    midpoint_12h = (bullish_ffilled + bearish_ffilled) / 2.0
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 34)  # Need 20 for volume MA, 35 for fractals (5+2), 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(midpoint_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 12h Williams fractals)
        breakout_up = close[i] > bearish_fractal_aligned[i]  # Break above bearish fractal (resistance)
        breakout_down = close[i] < bullish_fractal_aligned[i]  # Break below bullish fractal (support)
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 12h midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < midpoint_12h[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > midpoint_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0