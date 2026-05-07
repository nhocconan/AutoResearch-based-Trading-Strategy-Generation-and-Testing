#!/usr/bin/env python3
"""
6h_Williams_Fractal_Breakout_1wTrend_Volume
Hypothesis: Use daily Williams Fractals to identify support/resistance, break out with volume confirmation, and filter by weekly trend (EMA50). Enter long when price breaks above latest bearish fractal (resistance) with volume spike in weekly uptrend, short when price breaks below latest bullish fractal (support) with volume spike in weekly downtrend. Exit on opposite fractal break. Designed for 6h to capture multi-day swings with low frequency.
"""

name = "6h_Williams_Fractal_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Need 2 extra bars for fractal confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(
        prices, df_1w, ema_50_1w, additional_delay_bars=0
    )
    
    # Volume confirmation: 50-period average on 6h
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.divide(volume, vol_ma50, out=np.zeros_like(volume), where=vol_ma50!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend determination
        weekly_close_aligned = align_htf_to_ltf(
            prices, df_1w, df_1w['close'].values
        )
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_aligned[i] > ema_50_1w_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above bearish fractal (resistance) with volume spike in weekly uptrend
            if (close[i] > bearish_fractal_aligned[i] and
                vol_ratio[i] > 2.5 and 
                weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below bullish fractal (support) with volume spike in weekly downtrend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  vol_ratio[i] > 2.5 and 
                  weekly_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal (support)
            if close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal (resistance)
            if close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals