#!/usr/bin/env python3
# 1d Weekly Trend + Volume Spike with Dynamic Position Sizing
# Hypothesis: Weekly trend (via EMA10) filters direction, daily volume spikes (3x average) signal momentum, and volatility (ATR) adjusts position size to manage risk. Designed for low trade frequency (10-25/year) with strong performance in both bull and bear markets by avoiding overextended moves and catching institutional inflows.
name = "1d_WeeklyTrend_VolumeSpike_ATRSize"
timeframe = "1d"
leverage = 1.0

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
    
    # === Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_close_1w = df_1w['close'].values
    
    # Weekly EMA10 for trend filter
    ema_10_1w = pd.Series(weekly_close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # === Daily Volume Spike (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 3.0)
    
    # === Daily ATR (14-period) for Position Sizing ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_10_1d[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above weekly EMA10 (uptrend) + volume spike
            if (close[i] > ema_10_1d[i] and vol_spike[i]):
                # Size inversely proportional to volatility (ATR), capped at 0.30
                size = min(0.30, 0.015 / (atr[i] / close[i] + 0.001))  # Normalize ATR by price
                signals[i] = size
                position = 1
            # SHORT: Price below weekly EMA10 (downtrend) + volume spike
            elif (close[i] < ema_10_1d[i] and vol_spike[i]):
                size = min(0.30, 0.015 / (atr[i] / close[i] + 0.001))
                signals[i] = -size
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA10
            if close[i] <= ema_10_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA10
            if close[i] >= ema_10_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals