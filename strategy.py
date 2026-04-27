#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Fractal breakout with weekly trend filter and volume confirmation.
Goes long when bullish fractal forms above weekly EMA50 with volume > 2x average.
Goes short when bearish fractal forms below weekly EMA50 with volume > 2x average.
Uses discrete position sizes (±0.25) to minimize churn. Target: 15-30 trades/year.
Williams Fractals identify swing points; weekly EMA50 filters trend direction.
Works in bull/bear by capturing swing highs/lows in trending markets.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on weekly close
    close_1w = df_1w['close'].values
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to 12h timeframe (with 2-bar delay for fractal confirmation)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w, additional_delay_bars=2)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Align fractals to 12h timeframe (with 2-bar delay for confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation on 12h
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need weekly EMA (50), volume MA (20)
    start_idx = max(ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Get the daily close that corresponds to this 12h bar
        # We'll use the close from the most recent completed daily bar
        # Since we're on 12h timeframe, each 12h bar spans half a day
        # We approximate by using the current price vs daily EMA equivalent
        # For simplicity, we'll use the 12h close vs the aligned weekly EMA
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price vs weekly EMA
        above_weekly_ema = price > ema_1w_aligned[i]
        below_weekly_ema = price < ema_1w_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: bullish fractal AND price above weekly EMA AND volume confirmation
            if bullish_fractal_aligned[i] and above_weekly_ema and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: bearish fractal AND price below weekly EMA AND volume confirmation
            elif bearish_fractal_aligned[i] and below_weekly_ema and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA or bearish fractal forms
            if price < ema_1w_aligned[i] or bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above weekly EMA or bullish fractal forms
            if price > ema_1w_aligned[i] or bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsFractal_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0