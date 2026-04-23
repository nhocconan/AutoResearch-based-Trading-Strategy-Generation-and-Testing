#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
1w EMA50 provides strong trend filter to avoid counter-trend trades in ranging markets.
Donchian breakouts capture momentum while volume confirmation filters false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower bands: highest high/lowest low of last 20 periods
    high_rolling_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, high_rolling_max)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, low_rolling_min)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: 12h volume > 1.8x 20-period MA (balanced to reduce trades)
        vol_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > donchian_upper_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_lower_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: reverse signal or break of opposite Donchian band
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower band
                if close[i] < donchian_lower_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper band
                if close[i] > donchian_upper_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0