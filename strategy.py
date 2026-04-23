#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation.
Donchian channels identify medium-term price structure. Breakout above/below 20-period
high/low with 1w EMA200 trend and volume confirmation avoids false signals. Designed for
1d timeframe to capture sustained moves in both bull/bear markets. Target: 7-25 trades/year
(30-100 total over 4 years). Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 1d Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # need EMA200 and Donchian/vol MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA200 = uptrend, close < 1w EMA200 = downtrend
        trend_up = close[i] > ema_200_1w_aligned[i]
        trend_down = close[i] < ema_200_1w_aligned[i]
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > highest_20[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < lowest_20[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian band (lower for longs, upper for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower band
                if close[i] < lowest_20[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper band
                if close[i] > highest_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA200_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0