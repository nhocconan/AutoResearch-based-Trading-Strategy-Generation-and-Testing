#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation
- Long when price breaks above latest weekly Williams Bearish Fractal AND price > weekly EMA50 AND volume > 1.5x 20-period average
- Short when price breaks below latest weekly Williams Bullish Fractal AND price < weekly EMA50 AND volume > 1.5x 20-period average
- Exit when price crosses the weekly EMA50 (trend reversal signal)
- Uses Williams Fractals from weekly timeframe for significant swing points
- Weekly EMA50 for trend alignment to avoid counter-trend trades
- Volume confirmation reduces false breakouts
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag on 6h timeframe
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
    
    # Get weekly data for Williams Fractals and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly Williams Fractals (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Additional delay of 2 bars for fractal confirmation (needs 2 future weekly bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20)  # Need 52 for EMA50 (50+2), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > bearish_fractal_aligned[i]  # Break above weekly bearish fractal
        breakout_down = close[i] < bullish_fractal_aligned[i]  # Break below weekly bullish fractal
        
        # Trend filter
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
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
            # Exit: price crosses weekly EMA50 (trend reversal)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below weekly EMA50
                if close[i] < ema50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above weekly EMA50
                if close[i] > ema50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0