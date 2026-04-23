#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years). Uses discrete position sizing (0.30) to minimize fee churn.
Works in both bull/bear via 12h trend filter and volume confirmation to avoid false breakouts.
Donchian breakout provides clear structure with proven edge on SOLUSDT; adding 12h trend filter improves BTC/ETH performance.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 12h EMA50 = uptrend, close < 12h EMA50 = downtrend
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA (moderate to balance trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > high_roll[i] and trend_up and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short: Break below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < low_roll[i] and trend_down and vol_filter:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: reverse signal or break of opposite Donchian band
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower band
                if close[i] < low_roll[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper band
                if close[i] > high_roll[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0