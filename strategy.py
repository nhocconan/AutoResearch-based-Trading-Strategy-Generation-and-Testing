#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1wTrend_VolumeConfirmation
Hypothesis: 6h breakout of weekly Williams Fractal levels with 1w trend filter (price > weekly EMA34) and volume confirmation (>2.0x 20-bar MA). Uses weekly fractals for structure, 1w trend to avoid counter-trend trades, and volume to reduce false breakouts. Designed for 12-30 trades/year (50-120 total over 4 years) to minimize fee drag. Works in bull/bear markets by following weekly trend while using fractal breakouts for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend and fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly Williams Fractals (requires 2 extra bars for confirmation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1w, low_1w)
    # Align with 2 extra delay bars for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema, 2 for fractal delay)
    start_idx = max(20, 34, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1w = close_val > ema_34_val
        bearish_1w = close_val < ema_34_val
        
        # Entry conditions: breakout of weekly fractal levels in trend direction with volume spike
        long_entry = (close_val > bullish_fractal_val) and bullish_1w and vol_spike
        short_entry = (close_val < bearish_fractal_val) and bearish_1w and vol_spike
        
        # Exit conditions: opposite fractal level touch (bearish for long, bullish for short)
        exit_long = close_val < bearish_fractal_val
        exit_short = close_val > bullish_fractal_val
        
        # Minimum holding period: 3 bars (to avoid whipsaw)
        min_hold = 3
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0