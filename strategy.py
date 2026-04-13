#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
    # Long when price breaks above Camarilla H3 level and 1d EMA50 > EMA200 (uptrend) and volume > 1.5x average.
    # Short when price breaks below Camarilla L3 level and 1d EMA50 < EMA200 (downtrend) and volume > 1.5x average.
    # Exit when price crosses Camarilla pivot point (mean reversion to equilibrium).
    # Uses Camarilla pivot structure from 1d for key support/resistance levels.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Pivot + (Range * 1.1 / 4)
    # L3 = Pivot - (Range * 1.1 / 4)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    H3_1d = pivot_1d + (range_1d * 1.1 / 4.0)
    L3_1d = pivot_1d - (range_1d * 1.1 / 4.0)
    
    # Calculate EMA50 and EMA200 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average (20-period) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA50 > EMA200 for uptrend, < for downtrend
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > H3_1d_aligned[i]
        short_breakout = close[i] < L3_1d_aligned[i]
        
        # Exit conditions: price crosses pivot point (mean reversion)
        long_exit = close[i] < pivot_1d_aligned[i]
        short_exit = close[i] > pivot_1d_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and uptrend and volume_confirm and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and downtrend and volume_confirm and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0