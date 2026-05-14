#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly pivot direction and volume confirmation
# Williams Fractals identify key swing highs/lows. Breakout above recent bullish fractal or below bearish fractal with volume spike.
# Weekly pivot provides higher timeframe bias: long only above weekly pivot, short only below.
# Works in bull (breakouts with volume) and bear (mean reversion at extremes after volatility expansion).

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
    
    # Weekly HTF data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot points (using prior week to avoid look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to use only completed weekly bars
    pp_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r1_1w = 2 * pp_1w - np.roll(low_1w, 1)
    s1_1w = 2 * pp_1w - np.roll(high_1w, 1)
    r2_1w = pp_1w + (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    s2_1w = pp_1w - (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    r3_1w = np.roll(high_1w, 1) + 2 * (pp_1w - np.roll(low_1w, 1))
    s3_1w = np.roll(low_1w, 1) - 2 * (np.roll(high_1w, 1) - pp_1w)
    r4_1w = np.roll(high_1w, 1) + 3 * (pp_1w - np.roll(low_1w, 1))
    s4_1w = np.roll(low_1w, 1) - 3 * (np.roll(high_1w, 1) - pp_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Williams Fractals: 5-bar pattern (requires 2 bars on each side)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n-1] < low[n+1]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i-2] < high[i-1] and high[i-1] > high[i] and 
            high[i] > high[i+1] and high[i-1] > high[i+1]):
            bearish_fractal[i] = high[i]
        if (low[i-2] > low[i-1] and low[i-1] < low[i] and 
            low[i] < low[i+1] and low[i-1] < low[i+1]):
            bullish_fractal[i] = low[i]
    
    # Find most recent completed fractal (shift by 2 to avoid look-ahead)
    # We need to look back 2 bars to confirm the fractal is complete
    recent_bearish = np.full(n, np.nan)
    recent_bullish = np.full(n, np.nan)
    
    max_bearish = np.full(n, np.nan)
    min_bullish = np.full(n, np.nan)
    
    current_max_bear = np.nan
    current_min_bull = np.nan
    
    for i in range(n):
        if not np.isnan(bearish_fractal[i-2]):
            current_max_bear = bearish_fractal[i-2]
        if not np.isnan(bullish_fractal[i-2]):
            current_min_bull = bullish_fractal[i-2]
        recent_bearish[i] = current_max_bear
        recent_bullish[i] = current_min_bull
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for fractals and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(recent_bearish[i]) or 
            np.isnan(recent_bullish[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Weekly pivot bias
        above_weekly_pivot = curr_close > pp_1w_aligned[i]
        below_weekly_pivot = curr_close < pp_1w_aligned[i]
        
        # Fractal breakout conditions
        breakout_above_bearish = curr_close > recent_bearish[i]
        breakout_below_bullish = curr_close < recent_bullish[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above bearish fractal, volume spike, above weekly pivot
            if breakout_above_bearish and vol_spike and above_weekly_pivot:
                signals[i] = 0.25
                position = 1
            # Short: break below bullish fractal, volume spike, below weekly pivot
            elif breakout_below_bullish and vol_spike and below_weekly_pivot:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below bullish fractal or weekly pivot failure
            if curr_close < recent_bullish[i] or curr_close < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above bearish fractal or weekly pivot failure
            if curr_close > recent_bearish[i] or curr_close > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals