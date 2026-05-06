#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume spike
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit with signal=0 when price crosses 1d EMA50 in opposite direction
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian(20) provides clear structure breakout signals
# 1d EMA50 ensures higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms institutional participation and reduces false breakouts
# Works in bull via buying strength in uptrend breakouts, works in bear via selling strength in downtrend breakdowns

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: price breaks above Donchian high AND uptrend AND volume spike
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend reverses (price crosses below 1d EMA50)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend reverses (price crosses above 1d EMA50)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals