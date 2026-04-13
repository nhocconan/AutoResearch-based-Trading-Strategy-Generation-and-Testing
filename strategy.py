#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels (H4/L4) + 1w trend filter (EMA50) + volume confirmation.
# Long: Price touches or crosses above H4 level (from prior 1d) + price > weekly EMA50 + volume > 1.5x avg volume (20-period).
# Short: Price touches or crosses below L4 level (from prior 1d) + price < weekly EMA50 + volume > 1.5x avg volume.
# Uses 1d for pivot levels (key support/resistance), 1w for trend filter, 4h for execution.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 20-50 total trades over 4 years (5-12/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for Camarilla pivot levels (H4, L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: H4 = C + 1.1/2*(H-L), L4 = C - 1.1/2*(H-L)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_h4[i] = c + (1.1/2) * (h - l)
        camarilla_l4[i] = c - (1.1/2) * (h - l)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Camarilla levels to 4h (use prior day's levels)
    camarilla_h4_shifted = np.roll(camarilla_h4, 1)
    camarilla_l4_shifted = np.roll(camarilla_l4, 1)
    camarilla_h4_shifted[0] = np.nan  # First day has no prior
    camarilla_l4_shifted[0] = np.nan
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_shifted)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_shifted)
    
    # Align 1w EMA50 to 4h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price touches/above H4 + above weekly EMA50 + volume confirmation
            if (price >= h4_level and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price touches/below L4 + below weekly EMA50 + volume confirmation
            elif (price <= l4_level and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below L4 level or below weekly EMA50
            if (price <= l4_level or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above H4 level or above weekly EMA50
            if (price >= h4_level or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Camarilla_EMA_Volume"
timeframe = "4h"
leverage = 1.0