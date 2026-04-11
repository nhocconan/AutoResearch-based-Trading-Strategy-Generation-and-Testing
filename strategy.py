#!/usr/bin/env python3
# 4h_12h_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot breakout with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above R4 or below S4
# with volume confirmation and 12h trend alignment capture high-probability moves. Designed for low
# trade frequency (~20-40/year) to minimize fee drift. Works in bull markets via long breakouts and
# bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLC for Camarilla calculation
    high_4h = high
    low_4h = low
    close_4h = close
    
    # Daily OHLC from 12h data (approximate daily from 12h: use last 2 bars)
    # For Camarilla, we need daily high, low, close
    # We'll use 12h data and take every 2nd bar to approximate daily
    if len(df_12h) >= 2:
        # Resample 12h to daily by taking every 2nd bar (since 2*12h = 24h)
        daily_high = df_12h['high'].values[::2]
        daily_low = df_12h['low'].values[::2]
        daily_close = df_12h['close'].values[::2]
        
        # Ensure we have enough data
        if len(daily_high) < 2:
            return np.zeros(n)
    else:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    
    # We'll use H4 and L4 as breakout levels
    daily_range = daily_high - daily_low
    H4 = daily_close + 1.5 * daily_range
    L4 = daily_close - 1.5 * daily_range
    
    # Align daily Camarilla levels to 4h timeframe
    # Since we used every 2nd 12h bar for daily, we need to repeat each level twice
    H4_expanded = np.repeat(H4, 2)
    L4_expanded = np.repeat(L4, 2)
    
    # Trim or pad to match 12h data length
    if len(H4_expanded) > len(df_12h):
        H4_expanded = H4_expanded[:len(df_12h)]
        L4_expanded = L4_expanded[:len(df_12h)]
    elif len(H4_expanded) < len(df_12h):
        # Pad with last value
        pad_length = len(df_12h) - len(H4_expanded)
        H4_expanded = np.pad(H4_expanded, (0, pad_length), mode='edge')
        L4_expanded = np.pad(L4_expanded, (0, pad_length), mode='edge')
    
    H4_12h = H4_expanded
    L4_12h = L4_expanded
    
    # Align Camarilla levels from 12h to 4h
    H4_4h = align_htf_to_ltf(prices, df_12h, H4_12h)
    L4_4h = align_htf_to_ltf(prices, df_12h, L4_12h)
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(H4_4h[i]) or np.isnan(L4_4h[i]) or np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals
        breakout_up = high[i] > H4_4h[i-1]
        breakdown_down = low[i] < L4_4h[i-1]
        
        # 12h EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_20_12h_aligned[i]
        trend_bearish = close[i] < ema_20_12h_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H4 AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L4 AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout (breakdown for long, breakout for short)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals