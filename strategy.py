#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal + 1d EMA trend + volume spike
# Uses weekly fractals for reversal points with 1d EMA21 trend filter.
# Williams fractals require 2-bar confirmation after the pattern forms.
# Volume > 1.8x 20-period average confirms institutional participation.
# Targets 25-35 trades/year to minimize fee decay while capturing high-conviction reversals.
# Works in both bull/bear: fractals capture swing points, trend filter avoids counter-trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA21 for trend filter
    close_1d = df_1d['close'].values
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate Williams fractals on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    # Williams fractal: middle bar highest/lowest of 5 bars
    for i in range(2, len(high_1w) - 2):
        if (high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i-2] and
            high_1w[i] > high_1w[i+1] and high_1w[i] > high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]  # bearish fractal at high
        if (low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i-2] and
            low_1w[i] < low_1w[i+1] and low_1w[i] < low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]   # bullish fractal at low
    
    # Williams fractals need 2-bar confirmation after the pattern
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA21
        uptrend = price > ema_21_1d_aligned[i]
        downtrend = price < ema_21_1d_aligned[i]
        
        # Volume confirmation: spike > 1.8x average
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long at bullish fractal: price bounces from support in uptrend
            if uptrend and price <= bullish_fractal_aligned[i] * 1.001 and volume_confirmation:
                signals[i] = size
                position = 1
            # Short at bearish fractal: price rejects from resistance in downtrend
            elif downtrend and price >= bearish_fractal_aligned[i] * 0.999 and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches bearish fractal or breaks below EMA21
            if price >= bearish_fractal_aligned[i] * 0.999 or price < ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches bullish fractal or breaks above EMA21
            if price <= bullish_fractal_aligned[i] * 1.001 or price > ema_21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsFractal_EMA21_Trend_Volume"
timeframe = "6h"
leverage = 1.0