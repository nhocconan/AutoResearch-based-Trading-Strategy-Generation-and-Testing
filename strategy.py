#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-week Donchian breakout direction with 1-day volume confirmation and ATR-based volatility filter.
# Uses weekly Donchian(20) breakout for trend direction, enters on pullbacks to the 20-period EMA on 6h chart with volume spike.
# Volatility filter ensures trades only occur when ATR(14) is above its 50-period average, avoiding low-volatility chop.
# Designed for low trade frequency (15-25/year) to minimize fee impact while capturing high-probability trend continuations.

name = "6h_WeeklyDonchian_EMAPullback_VolumeVolatility"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian breakout direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly breakout signals: 1 for bullish breakout, -1 for bearish breakout
    weekly_breakout_up = np.where(high_1w > high_20, 1, 0)
    weekly_breakout_down = np.where(low_1w < low_20, -1, 0)
    weekly_direction = weekly_breakout_up + weekly_breakout_down  # 1, -1, or 0
    
    # Align weekly direction to 6h timeframe
    weekly_direction_aligned = align_htf_to_ltf(prices, df_1w, weekly_direction.astype(float))
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Daily volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ema_20 * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # 6h indicators: EMA(20) for pullback entries, ATR(14) for volatility filter
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr3[0] = 0  # First period has no prior close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR filter: current ATR > 1.2x 50-period EMA of ATR (avoid low volatility)
    atr_ema_50 = pd.Series(atr_14).ewm(span=50, adjust=False, min_periods=50).mean().values
    volatility_filter = atr_14 > (atr_ema_50 * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_direction_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: weekly bullish breakout, price near EMA(20), volume spike, adequate volatility
            if (weekly_direction_aligned[i] > 0.5 and  # Weekly bullish breakout
                close[i] <= ema_20[i] * 1.01 and       # Within 1% above EMA(20) (pullback)
                close[i] >= ema_20[i] * 0.99 and
                volume_spike_aligned[i] > 0.5 and      # Daily volume spike
                volatility_filter[i]):                 # Sufficient volatility
                signals[i] = 0.25
                position = 1
            # Short setup: weekly bearish breakout, price near EMA(20), volume spike, adequate volatility
            elif (weekly_direction_aligned[i] < -0.5 and  # Weekly bearish breakout
                  close[i] >= ema_20[i] * 0.99 and       # Within 1% below EMA(20) (pullback)
                  close[i] <= ema_20[i] * 1.01 and
                  volume_spike_aligned[i] > 0.5 and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly direction turns bearish or price breaks significantly above EMA
            if weekly_direction_aligned[i] < -0.5 or close[i] > ema_20[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly direction turns bullish or price breaks significantly below EMA
            if weekly_direction_aligned[i] > 0.5 or close[i] < ema_20[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals