#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Donchian channels capture volatility-based breakouts. Aligning with 1d EMA34 ensures we trade with the dominant
daily trend. Volume confirmation filters weak breakouts. Discrete position sizing (0.25) minimizes fee churn.
Designed to work in both bull and bear markets via 1d trend filter. Target: 12-37 trades/year per symbol.
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
    
    # Calculate 12h Donchian(20) - using 20 periods of 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower: 20-period high/low of 12h data
    donch_hi_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_lo_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe (12h)
    donch_hi_aligned = align_htf_to_ltf(prices, df_12h, donch_hi_12h)
    donch_lo_aligned = align_htf_to_ltf(prices, df_12h, donch_lo_12h)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # need Donchian20, EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND uptrend AND volume confirmation
            if close[i] > donch_hi_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND downtrend AND volume confirmation
            elif close[i] < donch_lo_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Donchian level (lower for longs, upper for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below Donchian lower
                if close[i] < donch_lo_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above Donchian upper
                if close[i] > donch_hi_aligned[i]:
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