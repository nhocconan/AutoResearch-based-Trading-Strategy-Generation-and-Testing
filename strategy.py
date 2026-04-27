#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1w trend filter and volume confirmation.
# In trending markets, price breaks beyond recent fractal highs/lows with continuation.
# Uses 1w EMA50 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on weekly close
    ema_50_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_50_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_50_1w[i-1]):
                ema_50_1w[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_50_1w[i] = close_1w[i] * alpha + ema_50_1w[i-1] * (1 - alpha)
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get weekly data for Williams Fractals (need 2 bars confirmation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams Fractals on weekly data
    bearish_fractal = np.zeros(len(df_1w), dtype=bool)
    bullish_fractal = np.zeros(len(df_1w), dtype=bool)
    
    for i in range(2, len(df_1w) - 2):
        # Bearish fractal: high[i] is highest among 5 bars (i-2 to i+2)
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest among 5 bars (i-2 to i+2)
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Williams fractals need 2 extra weekly bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
            trend_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            trend_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: bullish fractal breakout + uptrend + volume spike
            if (bullish_fractal_aligned[i] > 0 and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish fractal breakout + downtrend + volume spike
            elif (bearish_fractal_aligned[i] > 0 and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: trend turns down or opposite fractal appears
            if (not trend_up or 
                bearish_fractal_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns up or opposite fractal appears
            if (not trend_down or 
                bullish_fractal_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0