#!/usr/bin/env python3
# 6h_weekly_pivot_volume_confirmation_v1
# Hypothesis: 6h strategy using weekly pivot points for trend direction, volume confirmation (>1.5x 20-bar avg volume), and 1d EMA(50) for alignment. Enters long when price breaks above weekly R1 with volume and 1d uptrend, short when breaks below weekly S1 with volume and 1d downtrend. Exits on opposite pivot break. Weekly pivots provide structural support/resistance that works in both bull/bear markets, volume confirms conviction, and 1d EMA ensures alignment with higher timeframe trend. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA(50) for trend alignment
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot points from prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculations
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # HTF trend filter: price above/below 1d EMA(50)
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly S1
            if close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly R1
            if close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for weekly pivot breakout with volume and HTF confirmation
            bullish_breakout = (close[i] > r1_aligned[i]) and volume_confirmed and htf_uptrend
            bearish_breakout = (close[i] < s1_aligned[i]) and volume_confirmed and htf_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals