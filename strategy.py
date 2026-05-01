#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d weekly pivot direction and volume confirmation
# Uses Williams fractals from 1d to identify significant swing points, filtered by 1d weekly pivot bias.
# Volume spike ensures institutional participation. Designed to capture strong momentum moves 
# while avoiding chop. Target: 12-37 trades/year on 6h timeframe to minimize fee drag.
# Works in both bull and bear markets by using fractal breakouts with HTF directional filter.

name = "6h_WilliamsFractal_Breakout_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Williams fractals and weekly pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: bearish (sell) fractal = high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # bullish (buy) fractal = low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra bars for confirmation (completed fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d weekly pivot (using prior week's OHLC)
    # For simplicity, we use prior 1d bar's OHLC to approximate weekly bias
    # In practice, weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We'll use prior 1d bar as proxy for weekly direction
    weekly_pivot = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
    # We need prior bar's pivot to avoid look-ahead
    weekly_pivot_bias_bullish = close > weekly_pivot_aligned  # Will be shifted in alignment
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Weekly pivot bias from prior bar (already aligned and shifted)
        pivot_bullish = weekly_pivot_aligned[i]  # This is already the prior bar's value due to alignment
        pivot_bias_bullish = curr_close > pivot_bullish
        pivot_bias_bearish = curr_close < pivot_bullish
        
        # Fractal breakout conditions
        # Bullish fractal breakout: price breaks above recent bullish fractal (support)
        bullish_fractal_breakout = curr_close > bullish_fractal_aligned[i] and not np.isnan(bullish_fractal_aligned[i])
        # Bearish fractal breakout: price breaks below recent bearish fractal (resistance)
        bearish_fractal_breakout = curr_close < bearish_fractal_aligned[i] and not np.isnan(bearish_fractal_aligned[i])
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish fractal breakout, volume spike, bullish pivot bias
            if bullish_fractal_breakout and vol_spike and pivot_bias_bullish:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakout, volume spike, bearish pivot bias
            elif bearish_fractal_breakout and vol_spike and pivot_bias_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish fractal breakout or pivot bias turns bearish
            if bearish_fractal_breakout or not pivot_bias_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish fractal breakout or pivot bias turns bullish
            if bullish_fractal_breakout or not pivot_bias_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals