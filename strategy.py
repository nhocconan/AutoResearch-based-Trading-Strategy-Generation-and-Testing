#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 6-hour Williams Fractals (period=2) aligned with 1-day EMA200 trend.
In both bull and bear markets, fractal reversals in the direction of the higher timeframe
trend (1-day EMA200) capture high-probability swing points. Volume > 1.3x average confirms
momentum. Williams Fractals require 2-bar confirmation (center + 2 lower/highers),
so we apply additional_delay_bars=2 when aligning to avoid look-ahead. Target: 12-37 trades/year.
Position size: 0.25. Uses discrete levels to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close
    close_1d = df_1d['close'].values
    ema_200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])  # SMA seed
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] * multiplier) + (ema_200[i-1] * (1 - multiplier))
    
    # Align 1d EMA200 to 6h timeframe (waits for 1d bar close)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Williams Fractals on 1d data (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Apply additional_delay_bars=2 for confirmation (fractal needs 2 future bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 200 for EMA200 seed, 20 for volume, fractals need 5 bars (2+center+2)
    start_idx = max(200, vol_period, 5)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA200
        bullish = price > ema_200_aligned[i]
        bearish = price < ema_200_aligned[i]
        
        # Volume confirmation: > 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: bullish fractal (support) in bullish trend with volume
            if bullish_fractal_aligned[i] and bullish and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish fractal (resistance) in bearish trend with volume
            elif bearish_fractal_aligned[i] and bearish and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bearish fractal appears or trend turns bearish
            if bearish_fractal_aligned[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: bullish fractal appears or trend turns bullish
            if bullish_fractal_aligned[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0