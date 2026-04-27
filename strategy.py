#!/usr/bin/env python3
"""
6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm
Hypothesis: Uses weekly Williams fractals (bearish = sell signal, bullish = buy signal) to identify key reversal zones.
Enter long when price breaks above a bullish fractal AND 1d close > EMA50 (uptrend) AND volume confirmation.
Enter short when price breaks below a bearish fractal AND 1d close < EMA50 (downtrend) AND volume confirmation.
Exit when price returns to the fractal level or trend reverses. Designed for 6h timeframe to achieve 50-150 total trades over 4 years.
Williams fractals require 2-bar confirmation delay (aligned with additional_delay_bars=2) to avoid look-ahead.
Works in both bull and bear markets by following 1d trend while using fractals for precise entry/exit at swing points.
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
    
    # Get 1d data for trend filter and weekly for fractals
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly Williams fractals (require 2-bar confirmation after center bar)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    # Bearish fractal: high[ i ] > high[ i-1 ] and high[ i ] > high[ i+1 ] and high[ i-1 ] > high[ i-2 ] and high[ i+1 ] > high[ i+2 ]
    # Bullish fractal: low[ i ] < low[ i-1 ] and low[ i ] < low[ i+1 ] and low[ i-1 ] < low[ i-2 ] and low[ i+1 ] < low[ i+2 ]
    # Align with 2 extra bars delay for confirmation (fractal forms at close of center bar, confirmed after 2 more bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA50 (50), volume avg (20), and weekly data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_aligned[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of weekly fractal level with 1d trend filter AND volume
            # Long: price breaks above bullish fractal (resistance turned support) AND 1d uptrend AND volume
            long_condition = (close_val > bull_fractal) and (close_val > ema_val) and vol_conf
            # Short: price breaks below bearish fractal (support turned resistance) AND 1d downtrend AND volume
            short_condition = (close_val < bear_fractal) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to bullish fractal level OR trend breaks
            exit_condition = (close_val <= bull_fractal) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to bearish fractal level OR trend breaks
            exit_condition = (close_val >= bear_fractal) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0