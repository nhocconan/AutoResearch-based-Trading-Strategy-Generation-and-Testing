#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses Donchian channel breakouts from prior 20 periods for structure.
# 12h EMA50 for higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-bar avg) to reduce false breakouts.
# Discrete position sizing at ±0.25 to minimize fee drag.
# Session filter (08:00-20:00 UTC) to focus on liquid hours.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid overtrading.

name = "6h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_vals = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channel (20-period) from prior close to avoid look-ahead
    # Upper = max(high[-20:-1]), Lower = min(low[-20:-1])
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, lookback + 1)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, above 12h EMA50, volume confirmation
            if (curr_close > curr_donchian_upper and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, below 12h EMA50, volume confirmation
            elif (curr_close < curr_donchian_lower and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian lower (trailing stop via structure)
            if curr_close < curr_donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian upper (trailing stop via structure)
            if curr_close > curr_donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals