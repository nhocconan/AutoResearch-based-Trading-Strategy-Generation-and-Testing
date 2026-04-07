#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d volume confirmation
# In bear/ranging markets (2025+), price often reverts to the 4h mean after deviations.
# The 4h EMA(50) acts as dynamic support/resistance; we fade deviations >1 ATR.
# 1d volume filter ensures participation during accumulation/distribution phases.
# Session filter (08-20 UTC) reduces noise from low-liquidity periods.
# Target: 15-30 trades/year to minimize fee drag.

name = "1h_mean_reversion_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and mean
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend and mean reversion target
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # ATR for deviation measurement (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_1d.ewm(span=20, min_periods=20, adjust=False).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Deviation from 4h EMA in ATR units
        deviation = (close[i] - ema_4h_aligned[i]) / atr[i]
        
        # Volume filter: current 1d volume above average
        vol_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Mean reversion entry: fade deviations >1.5 ATR
        if position == 0:  # Flat, look for entry
            if deviation < -1.5 and vol_filter:  # Oversold -> long
                position = 1
                signals[i] = 0.20
            elif deviation > 1.5 and vol_filter:  # Overbought -> short
                position = -1
                signals[i] = -0.20
        elif position == 1:  # Long position
            # Exit when price returns to mean or reverses
            if deviation >= -0.5 or deviation > 1.5:  # Mean reversion or adverse move
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain
        elif position == -1:  # Short position
            # Exit when price returns to mean or reverses
            if deviation <= 0.5 or deviation < -1.5:  # Mean reversion or adverse move
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain
    
    return signals