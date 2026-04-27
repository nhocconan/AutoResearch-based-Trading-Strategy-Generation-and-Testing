#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakouts with weekly trend filter and volume confirmation
# Works in bull markets via breakout momentum, in bear via mean-reversion at band touches
# Weekly trend filter reduces whipsaws, volume confirms institutional interest
# Target: 20-50 trades/year on daily timeframe for low fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian Channel (20-period high/low)
    # Highest high over past 20 days
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over past 20 days
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian (20), weekly EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + weekly uptrend
            if close[i] > donchian_high and vol_spike_val and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume spike + weekly downtrend
            elif close[i] < donchian_low and vol_spike_val and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if close[i] < donchian_low or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if close[i] > donchian_high or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0