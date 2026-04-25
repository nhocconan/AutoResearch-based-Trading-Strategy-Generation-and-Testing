#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 20-bar Donchian high AND weekly pivot is bullish (close > weekly pivot) AND volume > 1.5x 20-bar average.
Short when price breaks below 20-bar Donchian low AND weekly pivot is bearish (close < weekly pivot) AND volume > 1.5x 20-bar average.
Exit via ATR trailing stop (2.0*ATR from extreme) or re-entry into Donchian channel.
Designed for ~12-37 trades/year by requiring strong breakouts, weekly pivot alignment, and volume confirmation.
Works in bull/bear markets via weekly pivot filter; avoids whipsaws via volume confirmation.
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
    
    # Get weekly data for pivot direction filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR for trailing stop (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # highest close since long entry
    short_low = 0.0   # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, lookback, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        weekly_pivot_val = weekly_pivot_aligned[i]
        
        if position == 0:
            # Only trade in alignment with weekly pivot direction
            if close[i] > weekly_pivot_val:  # Weekly bullish regime
                # Long: break above Donchian high with volume confirmation
                long_signal = (close[i] > highest_high[i]) and vol_regime[i]
            else:  # Weekly bearish regime
                # Short: break below Donchian low with volume confirmation
                short_signal = (close[i] < lowest_low[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_low = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest close
            if close[i] > long_high:
                long_high = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian channel
            atr_stop = long_high - 2.0 * atr[i]
            range_exit = (close[i] < highest_high[i] and close[i] > lowest_low[i])
            if close[i] <= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest close
            if close[i] < short_low:
                short_low = close[i]
            # Exit conditions: ATR trailing stop OR re-enter Donchian channel
            atr_stop = short_low + 2.0 * atr[i]
            range_exit = (close[i] > lowest_low[i] and close[i] < highest_high[i])
            if close[i] >= atr_stop or range_exit:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0