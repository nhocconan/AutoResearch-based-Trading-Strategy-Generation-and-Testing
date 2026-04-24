#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution and signal generation.
- HTF: 1d for EMA50 trend filter.
- Donchian levels calculated from prior 20 periods of 12h data.
- Long when price breaks above upper Donchian channel with volume spike in uptrend.
- Short when price breaks below lower Donchian channel with volume spike in downtrend.
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakouts in downtrend.
- Uses tight entry conditions to minimize fee drag and improve test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from 12h data (20-period lookback)
    # We need to calculate this on the 12h data itself, then align
    # For Donchian(20), we need high/low of last 20 periods
    # We'll calculate rolling max/min on 12h data
    
    # First, we need to resample our logic: since we're using 12h timeframe,
    # we should calculate Donchian on 12h bars directly
    # But we don't have 12h data as DataFrame, so we'll calculate from prices
    # using the fact that 12h bars are every 48th 15m bar (but we're on 12h TF)
    # Actually, since timeframe="12h", the prices DataFrame is already 12h data
    
    # Re-extract for clarity - prices is already 12h data
    high_12h = high
    low_12h = low
    close_12h = close
    volume_12h = volume
    
    # Calculate Donchian channels: 20-period high/low
    # Upper channel = max(high, lookback=20)
    # Lower channel = min(low, lookback=20)
    # We shift by 1 to avoid look-ahead (use prior 20 periods, not including current)
    lookback = 20
    if len(high_12h) < lookback + 1:
        return np.zeros(n)
        
    # Rolling window for Donchian - use prior 20 periods
    donchian_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback + 1, 50, 20)  # Donchian lookback + EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1d EMA50 trend
            if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
                ema50_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Long when price breaks above upper Donchian with volume spike
                    if close_12h[i] > donchian_high[i] and volume_spike[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Short when price breaks below lower Donchian with volume spike
                    if close_12h[i] < donchian_low[i] and volume_spike[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian channel
            if close_12h[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian channel
            if close_12h[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0