#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter.
Long when price breaks above 20-bar high AND volume > 1.5x 20-bar average AND ATR(14) > 0.5 * ATR(50).
Short when price breaks below 20-bar low AND volume > 1.5x 20-bar average AND ATR(14) > 0.5 * ATR(50).
Exit when price touches 20-bar midpoint or opposite Donchian level.
Uses 1d for trend regime (EMA50 slope) to avoid counter-trend trades in strong trends.
Designed to capture breakouts with volume confirmation in both ranging and trending markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 slope for trend regime (avoid counter-trend in strong moves)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.zeros_like(close_1d)
    ema50_slope[1:] = (ema50_1d[1:] - ema50_1d[:-1]) / ema50_1d[:-1]  # daily % change
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # Calculate 4h Donchian channels (20-bar)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h ATR for trend filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / (atr50 + 1e-10)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(ema50_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: avoid counter-trend trades in strong 1d trends
        # Only allow trades aligned with 1d EMA50 slope direction
        trend_up = ema50_slope_aligned[i] > 0.0005  # mild uptrend threshold
        trend_down = ema50_slope_aligned[i] < -0.0005  # mild downtrend threshold
        
        # Breakout conditions
        breakout_up = close[i] > donch_high[i]
        breakout_down = close[i] < donch_low[i]
        
        # Exit conditions: touch midpoint or opposite Donchian level
        touch_mid = abs(close[i] - donch_mid[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < donch_low[i]) or \
                         (position == -1 and close[i] > donch_high[i])
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and not strong downtrend
            if (breakout_up and volume_confirmed and not trend_down):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and not strong uptrend
            elif (breakout_down and volume_confirmed and not trend_up):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below Donchian low
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above Donchian high
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0