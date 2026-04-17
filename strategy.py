#!/usr/bin/env python3
"""
1h_4h_1d_SuperTrend_Bias_Momentum_v1
Hypothesis: Use 4h SuperTrend for trend direction, 1-day close-to-open for momentum bias, and 1-hour momentum for entry timing.
Works in bull/bear because SuperTrend adapts to volatility and daily bias filters counter-trend noise.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4-hour Data (HTF for trend direction via SuperTrend) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for SuperTrend
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr_period = 10
    atr = np.zeros_like(tr)
    for i in range(1, len(tr)):
        if i < atr_period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # SuperTrend calculation
    factor = 3.0
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + factor * atr
    lower = hl2 - factor * atr
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
            
        # Adjust bands
        if direction[i] == 1:
            if lower[i] < supertrend[i-1]:
                lower[i] = supertrend[i-1]
            supertrend[i] = lower[i]
        else:
            if upper[i] > supertrend[i-1]:
                upper[i] = supertrend[i-1]
            supertrend[i] = upper[i]
    
    # SuperTrend direction for trend filter
    supertrend_direction = direction
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction)
    
    # 4-hour Volume for Confirmation (20-period average)
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # === 1-day Data (HTF for momentum bias) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Daily momentum: close - open (positive = bullish, negative = bearish)
    daily_momentum = close_1d - open_1d
    # Normalize by average true range-like measure to avoid scale issues
    hl_range_1d = df_1d['high'].values - df_1d['low'].values
    avg_hl_1d = pd.Series(hl_range_1d).rolling(window=10, min_periods=10).mean().values
    # Avoid division by zero
    avg_hl_1d = np.where(avg_hl_1d == 0, 1, avg_hl_1d)
    daily_momentum_norm = daily_momentum / avg_hl_1d
    daily_momentum_norm_aligned = align_htf_to_ltf(prices, df_1d, daily_momentum_norm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(daily_momentum_norm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h bar's volume for confirmation
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_confirmed = vol_4h_current > 1.2 * vol_ma_4h_aligned[i]
        
        # 1-hour momentum: close vs prior close
        hour_momentum = close[i] - close[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: SuperTrend uptrend, daily bullish bias, hourly momentum up, volume confirmation
            if (supertrend_direction_aligned[i] == 1 and 
                daily_momentum_norm_aligned[i] > 0.1 and 
                hour_momentum > 0 and 
                vol_confirmed):
                signals[i] = 0.20
                position = 1
                continue
            # Short: SuperTrend downtrend, daily bearish bias, hourly momentum down, volume confirmation
            elif (supertrend_direction_aligned[i] == -1 and 
                  daily_momentum_norm_aligned[i] < -0.1 and 
                  hour_momentum < 0 and 
                  vol_confirmed):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit long: SuperTrend turns down OR momentum breaks down
            if (supertrend_direction_aligned[i] == -1 or 
                daily_momentum_norm_aligned[i] < -0.1 or 
                hour_momentum < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: SuperTrend turns up OR momentum breaks up
            if (supertrend_direction_aligned[i] == 1 or 
                daily_momentum_norm_aligned[i] > 0.1 or 
                hour_momentum > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_SuperTrend_Bias_Momentum_v1"
timeframe = "1h"
leverage = 1.0