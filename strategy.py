#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w trend filter and volume confirmation
# Long when price breaks above latest bullish fractal AND 1w close > 1w EMA34 (uptrend) AND volume > 1.5x 20 EMA
# Short when price breaks below latest bearish fractal AND 1w close < 1w EMA34 (downtrend) AND volume > 1.5x 20 EMA
# Uses 6h for entry timing, 1w for trend direction to avoid counter-trend trades.
# Discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Williams Fractals require 2-bar confirmation delay after the center bar.

name = "6h_WilliamsFractal_Breakout_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter and fractals - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1w = close_1w > ema_34_1w
    downtrend_1w = close_1w < ema_34_1w
    
    # Calculate Williams Fractals on 1w data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n_1w = len(high_1w)
    bearish_fractal = np.full(n_1w, np.nan)
    bullish_fractal = np.full(n_1w, np.nan)
    
    for i in range(2, n_1w - 2):
        # Bearish fractal (peak)
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and 
            high_1w[i-3] < high_1w[i-1] and 
            high_1w[i+1] < high_1w[i-1]):
            bearish_fractal[i-1] = high_1w[i-1]
        
        # Bullish fractal (trough)
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and 
            low_1w[i-3] > low_1w[i-1] and 
            low_1w[i+1] > low_1w[i-1]):
            bullish_fractal[i-1] = low_1w[i-1]
    
    # Align 1w trend and fractals to 6h timeframe with 2-bar confirmation delay for fractals
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(uptrend_1w_aligned[i]) or np.isnan(downtrend_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above latest bullish fractal AND 1w uptrend AND volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below latest bearish fractal AND 1w downtrend AND volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below latest bullish fractal OR 1w trend changes to downtrend
            if (close[i] < bullish_fractal_aligned[i] or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above latest bearish fractal OR 1w trend changes to uptrend
            if (close[i] > bearish_fractal_aligned[i] or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals