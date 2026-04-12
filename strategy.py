#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_1d_elder_ray_power_v1
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 1d timeframe.
# Trend filter: 12h EMA50 slope (rising/falling). Entry on 6h when power aligns with trend.
# Bull Power > 0 and rising + 12h EMA50 rising → long. Bear Power < 0 and falling + 12h EMA50 falling → short.
# Volume confirmation: current volume > 1.5x 20-period average.
# Designed to capture institutional moves in both bull and bear markets via power imbalance.
# Target: 15-30 trades/year per symbol for low friction.
name = "6h_12h_1d_elder_ray_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray to 6h timeframe (will be available after 1d bar closes)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # EMA50 slope: rising if current > previous, falling if current < previous
    ema50_slope = np.zeros_like(ema50_12h_aligned)
    ema50_slope[1:] = np.where(ema50_12h_aligned[1:] > ema50_12h_aligned[:-1], 1,
                               np.where(ema50_12h_aligned[1:] < ema50_12h_aligned[:-1], -1, 0))
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_slope[i])):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: Bull Power > 0 and rising, EMA50 slope rising
        if (bull_power_aligned[i] > 0 and 
            i > 0 and bull_power_aligned[i] > bull_power_aligned[i-1] and
            ema50_slope[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: Bear Power < 0 and falling, EMA50 slope falling
        elif (bear_power_aligned[i] < 0 and 
              i > 0 and bear_power_aligned[i] < bear_power_aligned[i-1] and
              ema50_slope[i] < 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite power signal or trend change
        elif ((bear_power_aligned[i] < 0 and position == 1) or
              (bull_power_aligned[i] > 0 and position == -1) or
              (position == 1 and ema50_slope[i] < 0) or
              (position == -1 and ema50_slope[i] > 0)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals