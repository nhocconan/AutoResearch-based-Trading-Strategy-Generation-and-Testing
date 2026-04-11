#!/usr/bin/env python3
# 6h_1w_1d_supertrend_v1
# Strategy: 6s Supertrend with weekly trend filter and daily volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Supertrend captures trends effectively. Weekly trend filter avoids counter-trend trades during major reversals.
# Daily volume confirms institutional participation. Designed to work in both bull (follow weekly uptrend) and bear (follow weekly downtrend) markets.
# Target: 15-35 trades/year to minimize fee drag while capturing major moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_supertrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Supertrend calculation on 6h data
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Volume confirmation: current 6h volume > 1.5x daily average volume (scaled to 6h)
    # Approximate: 6h volume vs daily volume - assuming 4x 6h bars per day
    vol_threshold = 1.5 * (vol_avg_20_1d_aligned / 4.0)  # Scale daily avg to per 6h bar
    vol_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: price above/below weekly EMA20
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Supertrend direction
        st_uptrend = direction[i] == 1
        st_downtrend = direction[i] == -1
        
        # Entry logic: Supertrend alignment + weekly trend + volume confirmation
        if st_uptrend and weekly_uptrend and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif st_downtrend and weekly_downtrend and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Supertrend reversal or weekly trend change
        elif position == 1 and (not st_uptrend or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not st_downtrend or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals