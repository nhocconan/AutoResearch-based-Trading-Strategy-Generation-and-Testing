#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirmation
Hypothesis: Williams fractal breaks on 6h chart, confirmed by 12h EMA50 trend and volume spike, capture momentum in both bull and bear markets. Uses discrete sizing (0.25) to limit drawdown and fees. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams fractals (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Williams fractals on 6h
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_6h['high'].values,
        df_6h['low'].values,
    )
    # Williams fractal needs 2 extra 6h bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_6h, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_6h, bullish_fractal, additional_delay_bars=2
    )
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need fractals (2 extra delay), EMA50 (50), volume avg (20)
    start_idx = max(2, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bearish_fract = bearish_fractal_aligned[i]
        bullish_fract = bullish_fractal_aligned[i]
        ema50 = ema50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price vs 12h EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend:
                # Long bias: long when bullish fractal breaks above with volume
                if (close_val > bullish_fract) and vol_conf:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend:
                # Short bias: short when bearish fractal breaks below with volume
                if (close_val < bearish_fract) and vol_conf:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price retests broken fractal level or opposite fractal
            if close_val <= bullish_fract or close_val >= bearish_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price retests broken fractal level or opposite fractal
            if close_val >= bearish_fract or close_val <= bullish_fract:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_12hTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0