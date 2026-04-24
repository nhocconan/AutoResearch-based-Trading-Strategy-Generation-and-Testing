#!/usr/bin/env python3
"""
Hypothesis: 1h EMA crossover with 4h trend filter and 1d volume spike for timing.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA trend direction, 1d for volume average confirmation.
- Entry: Long when 1h EMA(9) crosses above EMA(21) AND 4h EMA(50) is rising (uptrend) AND volume > 1.5 * 1d average volume.
         Short when 1h EMA(9) crosses below EMA(21) AND 4h EMA(50) is falling (downtrend) AND volume > 1.5 * 1d average volume.
- Exit: Opposite 1h EMA crossover.
- Signal size: 0.20 discrete to minimize fee drag.
- Uses EMA crossover for momentum timing with 4h trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakout legitimacy.
- Designed to work in both bull and bear markets by following the 4h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA(50)
        return np.zeros(n)
    
    ema_50_4h = ema(df_4h['close'].values, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1h EMA(9) and EMA(21) for entry timing
    ema_9 = ema(close, 9)
    ema_21 = ema(close, 21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(9, 21, 50, 20)  # Need 9/21 for EMA, 50 for 4h EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_9[i]) or np.isnan(ema_21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_ema_9 = ema_9[i-1]
        prev_ema_21 = ema_21[i-1]
        
        # Exit conditions: opposite 1h EMA crossover
        if position != 0:
            # Exit long: EMA(9) crosses below EMA(21)
            if position == 1:
                if ema_9[i] < ema_21[i] and prev_ema_9 >= prev_ema_21:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: EMA(9) crosses above EMA(21)
            elif position == -1:
                if ema_9[i] > ema_21[i] and prev_ema_9 <= prev_ema_21:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: 1h EMA crossover with 4h trend filter and volume confirmation
        if position == 0:
            # 1h EMA crossover signals
            cross_up = ema_9[i] > ema_21[i] and prev_ema_9 <= prev_ema_21
            cross_down = ema_9[i] < ema_21[i] and prev_ema_9 >= prev_ema_21
            
            # 4h trend filter: EMA(50) rising for long, falling for short
            # Use previous bar to avoid look-ahead (trend based on completed 4h bar)
            if i > 0:
                ema_50_prev = ema_50_4h_aligned[i-1]
                ema_50_curr = ema_50_4h_aligned[i]
                trend_rising = ema_50_curr > ema_50_prev
                trend_falling = ema_50_curr < ema_50_prev
            else:
                trend_rising = False
                trend_falling = False
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if cross_up and trend_rising and volume_confirm:
                signals[i] = 0.20
                position = 1
            elif cross_down and trend_falling and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_EMA9_21_Crossover_4hEMA50_Trend_1dVolume_v1"
timeframe = "1h"
leverage = 1.0