#!/usr/bin/env python3
"""
1d_WilliamsFractal_Breakout_1wTrend_Volume
Hypothesis: Williams Fractals identify swing highs/lows on daily chart. Breakouts above recent bullish fractal or below bearish fractal with weekly trend filter (EMA34) and volume confirmation capture momentum moves. Works in bull (breakouts above fractals in uptrend) and bear (breakdowns below fractals in downtrend). Target: 20-60 total trades over 4 years (5-15/year).
"""

name = "1d_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    # Williams Fractals on daily data (5-bar window: 2 left, 2 right)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values if 'df_1d' in locals() else high,  # We'll get df_1d below
        df_1d['low'].values if 'df_1d' in locals() else low
    )
    # Get daily data for fractals
    df_1d = get_htf_data(prices, '1d')
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Need 2-bar confirmation delay for fractals (they form after 2 bars to the right)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for weekly indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or \
           np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average weekly volume (scaled to daily)
        # Approximate daily volume from weekly: weekly volume / 5 (5 trading days per week)
        vol_daily_approx = vol_sma20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_daily_approx
        
        if position == 0:
            # Long: Price breaks above recent bullish fractal with uptrend and volume
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below recent bearish fractal with downtrend and volume
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below bullish fractal or trend turns down
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above bearish fractal or trend turns up
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals