#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h trading using 1-week high/low breakout with volume confirmation and ADX trend filter
# Weekly high/low from 1w data provides key institutional levels on longer timeframe
# Breakout above weekly high or below weekly low captures strong momentum moves
# Volume > 1.5x average confirms institutional participation
# ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions
# Works in both bull and bear markets as breakouts occur in both directions
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Load 1d data ONCE for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly high/low from prior week (using 1w data)
    # Weekly high/low from previous completed week
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get prior week's high/low (excluding current incomplete week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Align weekly high/low to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # Calculate ADX (14) on 1d for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = 100 * wilders_smoothing(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilders_smoothing(minus_dm, 14) / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need enough for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: break above weekly high + volume + trending
            if (close[i] > weekly_high_aligned[i] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: break below weekly low + volume + trending
            elif (close[i] < weekly_low_aligned[i] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly low or breaks below it
            if close[i] < weekly_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly high or breaks above it
            if close[i] > weekly_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_HighLow_Breakout_Volume_ADX_v1"
timeframe = "12h"
leverage = 1.0