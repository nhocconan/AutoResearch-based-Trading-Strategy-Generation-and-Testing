#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal Breakout + 1w EMA50 Trend + Volume Spike Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for major trend filter (price > EMA50 = bullish regime, price < EMA50 = bearish regime).
- Entry: Long when price breaks above latest Williams bearish fractal AND price > 1w EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when price breaks below latest Williams bullish fractal AND price < 1w EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when price crosses below latest bullish fractal; Short exits when price crosses above latest bearish fractal.
- Signal size: 0.25 discrete to control fee drag.
- Uses Williams fractals for precise swing high/low breakouts, weekly EMA for regime filter, volume confirmation for participation.
- Williams fractals require 2-bar confirmation delay via align_htf_to_ltf(additional_delay_bars=2).
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1w data for Williams fractals (requires 2-bar confirmation delay)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    
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
                # Long: Close breaks above bearish fractal (swing high) AND price > 1w EMA50 (bullish regime)
                if curr_close > bearish_fractal_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below bullish fractal (swing low) AND price < 1w EMA50 (bearish regime)
                elif curr_close < bullish_fractal_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below bullish fractal (swing low)
            if curr_close < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above bearish fractal (swing high)
            if curr_close > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0