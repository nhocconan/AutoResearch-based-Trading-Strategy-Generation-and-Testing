#!/usr/bin/env python3
"""
6h_WilliamsFractal_Trend_Reversal
Hypothesis: On 6-hour timeframe, use daily Williams Fractal reversals with 1-day trend filter and volume confirmation. Fractals identify potential turning points after exhaustion moves; trading in the direction of the daily trend (via 21 EMA) avoids counter-trend traps. Volume surge confirms institutional participation at reversal points. Designed for low-moderate trade frequency (~20-40/year) to capture meaningful reversals while minimizing fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and fractals
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily 21 EMA for trend filter
    close_daily = df_daily['close'].values
    ema21_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily EMA to 6h timeframe
    ema21_daily_aligned = align_ltf_to_htf(prices, df_daily, ema21_daily)
    
    # Daily trend: bullish when price > EMA21
    daily_uptrend = close_daily > ema21_daily
    daily_downtrend = close_daily < ema21_daily
    
    # Align daily trend to 6h timeframe (with 1-bar delay for completed daily candle)
    daily_uptrend_aligned = align_ltf_to_htf(prices, df_daily, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_ltf_to_htf(prices, df_daily, daily_downtrend.astype(float))
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_daily['high'].values,
        df_daily['low'].values,
    )
    
    # Align fractals to 6h timeframe with 2-bar delay for confirmation
    bearish_fractal_aligned = align_ltf_to_htf(
        prices, df_daily, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_ltf_to_htf(
        prices, df_daily, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume confirmation: current volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_surge = volume > (vol_ma_50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_daily_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: fractal reversal in direction of daily trend with volume surge
        long_entry = bullish_fractal_aligned[i] and daily_uptrend_aligned[i] and volume_surge[i]
        short_entry = bearish_fractal_aligned[i] and daily_downtrend_aligned[i] and volume_surge[i]
        
        # Exit on opposite fractal with volume surge
        long_exit = bearish_fractal_aligned[i] and volume_surge[i]
        short_exit = bullish_fractal_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsFractal_Trend_Reversal"
timeframe = "6h"
leverage = 1.0