#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period high AND close > 1d EMA50 (uptrend) AND volume > 2.0x 20-period MA.
Short when price breaks below 20-period low AND close < 1d EMA50 (downtrend) AND volume > 2.0x 20-period MA.
Exit when price returns to midpoint of Donchian channel or opposite breakout occurs.
Designed for ~15-25 trades/year on 12h timeframe with proven edge from DB top performers.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channel (20-period)
    period = 20
    # For 12h timeframe, we need to calculate on 12h data
    # Since we're generating 12h signals, prices is already 12h data
    highest_20 = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_20 = pd.Series(low).rolling(window=period, min_periods=period).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, period, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(midpoint_20[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA50 = uptrend, close < 1d EMA50 = downtrend
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]  # Break above 20-period high
        breakout_down = close[i] < lowest_20[i]  # Break below 20-period low
        return_to_mid = abs(close[i] - midpoint_20[i]) < (highest_20[i] - lowest_20[i]) * 0.1  # Within 10% of midpoint
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above highest_20 AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lowest_20 AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to midpoint or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_mid or opposite_extreme
            elif position == -1:
                exit_signal = return_to_mid or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0