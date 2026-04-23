#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d close > 1d EMA34 AND volume > 1.5x 24-period MA.
Short when price breaks below Donchian lower band AND 1d close < 1d EMA34 AND volume > 1.5x 24-period MA.
Exit when price crosses Donchian midline (20-period average of high/low).
Designed for low trade frequency (target: 12-37/year) with trend following in 12h timeframe.
Donchian channels provide clear breakout signals, daily trend filter ensures alignment with higher timeframe momentum.
Volume confirmation reduces false breakouts. Strategy should work in both bull and bear markets by
trading breakouts in the direction of the daily trend.
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
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (24-period) for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24, 20)  # need EMA34, volume MA24, and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 24-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_24[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend AND volume filter
            if close[i] > highest_high[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume filter
            elif close[i] < lowest_low[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Donchian midline
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below Donchian midline
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above Donchian midline
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0