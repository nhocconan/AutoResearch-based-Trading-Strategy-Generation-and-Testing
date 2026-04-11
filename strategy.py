#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_v1
# Strategy: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels on daily chart act as strong support/resistance.
# Breakouts with weekly EMA trend alignment and volume capture significant moves.
# Designed for low trade frequency (~10-25/year) to avoid fee drag and work in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Using close of previous day as base
    prev_close = close[np.roll(np.arange(n), 1)]  # Previous day's close
    prev_close[0] = np.nan  # First day has no previous
    prev_high = high[np.roll(np.arange(n), 1)]    # Previous day's high
    prev_high[0] = np.nan
    prev_low = low[np.roll(np.arange(n), 1)]      # Previous day's low
    prev_low[0] = np.nan
    
    # Camarilla levels: H3, L3 (primary levels for breakout)
    # Range = prev_high - prev_low
    # H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    # L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    
    # 20-period EMA on weekly for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Camarilla levels
        # Breakout above H3 (strong resistance)
        breakout_up = high[i] > H3[i-1]
        # Breakdown below L3 (strong support)
        breakdown_down = low[i] < L3[i-1]
        
        # Weekly EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_20_1w_aligned[i]
        trend_bearish = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: Breakout above H3 AND bullish trend AND volume confirmation
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND bearish trend AND volume confirmation
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using H3/L3 levels (reverse the breakout)
        elif position == 1 and low[i] < L3[i-1]:  # Break back below L3
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H3[i-1]:  # Break back above H3
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals