#!/usr/bin/env python3
"""
1h_4h_1d_CloseBased_Momentum_With_Filter_v1
1-hour strategy using 4-hour trend direction (EMA34) and 1-day momentum (close-to-open) for bias, with 1-hour entry on momentum continuation.
Enters long when 4h EMA34 trending up, 1-day bullish bias, and 1-hour close > prior close (momentum continuation).
Enters short when 4h EMA34 trending down, 1-day bearish bias, and 1-hour close < prior close (momentum continuation).
Uses session filter (08-20 UTC) and volume confirmation to reduce false signals.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4-hour Data (HTF for trend direction) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4-hour EMA34 for Trend Filter (direction only)
    ema34_4h = pd.Series(close_4h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_slope = ema34_4h[1:] - ema34_4h[:-1]  # positive = up, negative = down
    ema34_4h_slope = np.concatenate([[0], ema34_4h_slope])  # align length
    ema34_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h_slope)
    
    # 4-hour Volume for Confirmation (20-period average)
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
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_4h_slope_aligned[i]) or 
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
            # Long: 4h trend up, daily bullish bias, hourly momentum up, volume confirmation
            if (ema34_4h_slope_aligned[i] > 0 and 
                daily_momentum_norm_aligned[i] > 0.1 and 
                hour_momentum > 0 and 
                vol_confirmed):
                signals[i] = 0.20
                position = 1
                continue
            # Short: 4h trend down, daily bearish bias, hourly momentum down, volume confirmation
            elif (ema34_4h_slope_aligned[i] < 0 and 
                  daily_momentum_norm_aligned[i] < -0.1 and 
                  hour_momentum < 0 and 
                  vol_confirmed):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: reverse signal or loss of momentum
        elif position == 1:
            # Exit long: trend turns down OR momentum breaks down
            if (ema34_4h_slope_aligned[i] < 0 or 
                daily_momentum_norm_aligned[i] < -0.1 or 
                hour_momentum < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend turns up OR momentum breaks up
            if (ema34_4h_slope_aligned[i] > 0 or 
                daily_momentum_norm_aligned[i] > 0.1 or 
                hour_momentum > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_CloseBased_Momentum_With_Filter_v1"
timeframe = "1h"
leverage = 1.0