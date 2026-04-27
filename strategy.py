#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volume spike and trend filter.
# Uses Donchian channel (20-period) on 4h prices: long when price breaks above upper band with 1-day uptrend and volume spike,
# short when breaks below lower band with 1-day downtrend and volume spike.
# Volume filter: current volume > 2.5x 24-period average (approx 4 days of 4h bars).
# Trend filter: 1-day EMA50 slope (rising/falling over 2 periods).
# Designed for 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drift.
# Works in both bull and bear markets by following the 1-day trend and requiring volatility expansion.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1-day data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need enough for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 50-period EMA on 1-day close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: rising if current > previous, falling if current < previous
    ema50_slope = np.diff(ema50_1d, prepend=ema50_1d[0])
    
    # Align 1-day EMA and slope to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_slope)
    
    # Volume filter: volume > 2.5x 24-period average (approx 4 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above upper Donchian band AND 1-day uptrend AND volume spike
        if (close[i] > highest_high[i] and 
            ema50_slope_aligned[i] > 0 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below lower Donchian band AND 1-day downtrend AND volume spike
        elif (close[i] < lowest_low[i] and 
              ema50_slope_aligned[i] < 0 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dEMA50Slope_VolumeFilter"
timeframe = "4h"
leverage = 1.0