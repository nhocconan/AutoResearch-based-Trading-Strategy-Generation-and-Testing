#!/usr/bin/env python3
"""
6h_WilliamsFractal_Trend_VolumeConfirm_v1
Hypothesis: Williams Fractals on 1d combined with 12h EMA50 trend filter and volume confirmation (>2.0x average) captures reliable reversal points at swing highs/lows. Works in both bull and bear markets by trading in direction of higher timeframe trend. Uses discrete sizing (0.25) and close-based exits when opposite fractal forms. Target 12-25 trades/year to minimize fee drag while maintaining edge.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter (aligned with extra delay for completed bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams Fractals on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Align with 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 1 day)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of EMA(50), volume(24), fractal calculation
    start_idx = max(50, 24, 10)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        bull_fract = bullish_fractal_aligned[i]
        bear_fract = bearish_fractal_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or 
            np.isnan(bull_fract) or np.isnan(bear_fract)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long: bullish fractal forms (swing low) with 12h uptrend and volume
        long_condition = (bull_fract == 1) and (close_val > ema_val) and volume_confirmed
        # Short: bearish fractal forms (swing high) with 12h downtrend and volume
        short_condition = (bear_fract == 1) and (close_val < ema_val) and volume_confirmed
        
        # Exit: opposite fractal forms (potential reversal)
        long_exit = (position == 1 and bear_fract == 1)
        short_exit = (position == -1 and bull_fract == 1)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_WilliamsFractal_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0