#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal Breakout + 12h EMA50 Trend + Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above a confirmed Williams bearish fractal AND price > 12h EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below a confirmed Williams bullish fractal AND price < 12h EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when price crosses below the most recent bullish fractal; Short exits when price crosses above the most recent bearish fractal.
- Signal size: 0.25 discrete to control fee drag.
- Uses Williams fractals for precise S/R from swing points, volume confirmation for participation,
  EMA50 trend filter to avoid counter-trend trades. Fractals require 2-bar confirmation (additional_delay_bars=2).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams fractals (prior completed 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Williams fractals on 6h
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_6h, low_6h)
    # Align with 2 extra bars for confirmation (fractals need 2 future candles to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_6h, bullish_fractal, additional_delay_bars=2)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above bearish fractal AND price > 12h EMA50 (uptrend)
                if curr_close > bearish_fractal_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below bullish fractal AND price < 12h EMA50 (downtrend)
                elif curr_close < bullish_fractal_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below bullish fractal
            if curr_close < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above bearish fractal
            if curr_close > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0