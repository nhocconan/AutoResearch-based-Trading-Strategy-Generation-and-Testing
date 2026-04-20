#!/usr/bin/env python3
# 1d_1W_TurtleSystem_Strategy_v1
# Hypothesis: 1d Turtle Trading System (20-day breakout) with 1w trend filter and volume confirmation
# captures strong trends in both bull and bear markets. Uses ATR-based position sizing and stops.
# Designed for low trade frequency (10-25/year) to minimize fee drag on higher timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_TurtleSystem_Strategy_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly trend filter: 50-period EMA ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-day Donchian channels (entry signals)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day ATR for position sizing and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current vs 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_atr = 0.0  # ATR at entry for stop calculation
    
    for i in range(20, n):  # Start after Donchian warmup
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema_50_val = ema_50_1w_aligned[i]
        atr_val = atr_10[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_20_val) or np.isnan(low_20_val) or 
            np.isnan(ema_50_val) or np.isnan(atr_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above 20-day high + weekly uptrend + volume
            if (close_val > high_20_val and 
                close_val > ema_50_val and 
                vol_ratio_val > 2.0):
                # Position size: 1 unit = 1% of volatility (ATR-based)
                # Scale to 0.25-0.35 range based on volatility
                vol_factor = min(2.0, max(0.5, 1.0 / (atr_val / close_val * 100)))
                size = 0.30 * vol_factor
                size = max(0.20, min(0.35, size))  # Clamp to reasonable range
                signals[i] = size
                position = 1
                entry_atr = atr_val
            # Short entry: Price breaks below 20-day low + weekly downtrend + volume
            elif (close_val < low_20_val and 
                  close_val < ema_50_val and 
                  vol_ratio_val > 2.0):
                vol_factor = min(2.0, max(0.5, 1.0 / (atr_val / close_val * 100)))
                size = 0.30 * vol_factor
                size = max(0.20, min(0.35, size))
                signals[i] = -size
                position = -1
                entry_atr = atr_val
        
        elif position == 1:
            # Long exit: Price drops to 10-day low OR 2*ATR trailing stop
            exit_condition = (close_val <= low_20[i]) or \
                           (close_val <= high_val - 2.0 * entry_atr)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # Maintain position
        
        elif position == -1:
            # Short exit: Price rises to 10-day high OR 2*ATR trailing stop
            exit_condition = (close_val >= high_20[i]) or \
                           (close_val >= low_val + 2.0 * entry_atr)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # Maintain position
    
    return signals