#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# - Donchian breakout: price > upper channel (20-period high) for long, < lower channel (20-period low) for short
# - 1d EMA50 filter: only trade long when price > EMA50, short when price < EMA50 to avoid counter-trend
# - Volume confirmation: 6h volume > 1.5x 20-period average to confirm breakout strength
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Position size: 0.25 (discrete levels to reduce churn)
# - Stoploss: exit when Donchian channel reverses (price < upper channel for long, > lower channel for short)

name = "6h_1d_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) channels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Upper channel: 20-period high
    upper_channel = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # 6h volume confirmation: > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below upper channel (failed breakout)
            if close_6h[i] < upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above lower channel (failed breakdown)
            if close_6h[i] > lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long signal: price breaks above upper channel in 1d uptrend
                if close_6h[i] > upper_channel[i] and close_6h[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: price breaks below lower channel in 1d downtrend
                elif close_6h[i] < lower_channel[i] and close_6h[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals