#!/usr/bin/env python3
"""
4h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm_HTF
Hypothesis: Uses daily Williams Fractals for breakout entries on 4h timeframe.
Enter long when price breaks above the most recent bullish fractal AND 1d EMA50 > EMA200 (uptrend) AND volume > 1.5 * 20-period average.
Enter short when price breaks below the most recent bearish fractal AND 1d EMA50 < EMA200 (downtrend) AND volume > 1.5 * 20-period average.
Exit when price returns to the opposite fractal level OR trend reverses.
Williams Fractals identify key swing highs/lows; daily trend filter ensures alignment with higher timeframe structure.
High volume threshold (1.5x) filters weak breakouts. Target: 75-150 total trades over 4 years (19-37/year) with 0.25 position size.
Designed to work in both bull and bear markets via trend filter and breakout logic.
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
    
    # Get 1d data for Williams Fractals and daily trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 and EMA200 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA200 (200), volume avg (20), fractal delay (2)
    start_idx = max(200, 20, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Williams Fractal levels with 1d trend filter AND volume
            # Long: price breaks above bullish fractal AND 1d uptrend AND volume
            long_condition = (close_val > bullish_fractal_val) and (ema_50_val > ema_200_val) and vol_conf
            # Short: price breaks below bearish fractal AND 1d downtrend AND volume
            short_condition = (close_val < bearish_fractal_val) and (ema_50_val < ema_200_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to bearish fractal level OR trend breaks
            exit_condition = (close_val <= bearish_fractal_val) or (ema_50_val <= ema_200_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to bullish fractal level OR trend breaks
            exit_condition = (close_val >= bullish_fractal_val) or (ema_50_val >= ema_200_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm_HTF"
timeframe = "4h"
leverage = 1.0