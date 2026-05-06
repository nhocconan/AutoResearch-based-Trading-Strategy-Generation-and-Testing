#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams Fractal breakout with volume confirmation and 1-week trend filter
# Long when price breaks above previous day's bearish fractal high with volume > 1.3x 20-period average
# Short when price breaks below previous day's bullish fractal low with volume > 1.3x 20-period average
# Uses 1-week EMA50 as trend filter: only long when price > EMA50, only short when price < EMA50
# Williams Fractals identify key swing points; breakouts with volume confirm institutional interest
# Trend filter prevents counter-trend trades in strong moves
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dWilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 5-bar window: 2 left, center, 2 right)
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    
    bearish_fractal = np.full(len(df_1d), np.nan)  # High at fractal point
    bullish_fractal = np.full(len(df_1d), np.nan)  # Low at fractal point
    
    # Williams Fractal: bar is highest/lowest of 5 bars (2 left, 2 right)
    for i in range(2, len(df_1d) - 2):
        # Bearish fractal: highest high
        if (high_arr[i] >= high_arr[i-2] and high_arr[i] >= high_arr[i-1] and
            high_arr[i] >= high_arr[i+1] and high_arr[i] >= high_arr[i+2]):
            bearish_fractal[i] = high_arr[i]
        # Bullish fractal: lowest low
        if (low_arr[i] <= low_arr[i-2] and low_arr[i] <= low_arr[i-1] and
            low_arr[i] <= low_arr[i+1] and low_arr[i] <= low_arr[i+2]):
            bullish_fractal[i] = low_arr[i]
    
    # Need 2 additional bars for fractal confirmation (per rule 2b)
    bearish_fractal_confirmed = np.roll(bearish_fractal, 2)
    bullish_fractal_confirmed = np.roll(bullish_fractal, 2)
    # First 2 bars cannot be confirmed
    bearish_fractal_confirmed[:2] = np.nan
    bullish_fractal_confirmed[:2] = np.nan
    
    # Align fractal levels to 4h timeframe with 2-bar confirmation delay
    bearish_fractal_4h = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed)
    bullish_fractal_4h = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed)
    
    # Get 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bearish_fractal_4h[i]) or np.isnan(bullish_fractal_4h[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above bearish fractal (resistance) with volume and trend filter
            if (close[i] > bearish_fractal_4h[i] and volume_filter[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price below bullish fractal (support) with volume and trend filter
            elif (close[i] < bullish_fractal_4h[i] and volume_filter[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal (support break)
            if close[i] < bullish_fractal_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal (resistance break)
            if close[i] > bearish_fractal_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals