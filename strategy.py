#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d ADX trend filter and volume spike confirmation
- Williams %R(14): Long when crosses above -80 from below (oversold bounce)
                    Short when crosses below -20 from above (overbought rejection)
- 1d ADX(14): Trend filter - only trade when ADX > 25 (strong trend present)
- Volume confirmation: Current volume > 2.0x 20-period average (institutional participation)
- Exit: Williams %R returns to opposite extreme zone (> -20 for longs, < -80 for shorts)
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag
- Williams %R identifies momentum extremes; works in both bull/bear markets by fading exhaustion
  while ADX ensures we only trade when there's sufficient trend strength to follow through
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (similar to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Williams %R on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # ADX needs ~50, volume MA needs 20, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or
            highest_high[i] == lowest_low[i]):  # Avoid division by zero artifacts
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d ADX
        strong_trend = adx_aligned[i] > 25
        
        # Williams %R signals with trend filter and volume confirmation
        # Long: Williams %R crosses above -80 from below (oversold bounce) + strong trend + volume spike
        # Short: Williams %R crosses below -20 from above (overbought rejection) + strong trend + volume spike
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r[i]
        
        long_signal = (williams_r_prev <= -80 and 
                      williams_r[i] > -80 and
                      strong_trend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r_prev >= -20 and 
                       williams_r[i] < -20 and
                       strong_trend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to opposite extreme zone
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns above -20 (overbought territory)
                if williams_r[i] >= -20:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns below -80 (oversold territory)
                if williams_r[i] <= -80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0