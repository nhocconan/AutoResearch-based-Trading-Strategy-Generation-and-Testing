#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with weekly trend filter (1w EMA50) and volume confirmation (>1.5x 20-period average)
    # Only trade breakouts in direction of weekly trend to avoid counter-trend whipsaws
    # Weekly trend filter works in both bull/bear markets by aligning with dominant 1w trend
    # Volume confirmation ensures institutional participation
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume MA for confirmation (>1.5x 20-period average)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-20:i])
    volume_spike_1w = volume_1w > (1.5 * vol_ma_1w)
    
    # Calculate 6h Donchian(20) channels
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        donchian_high[i] = np.max(high[i-donchian_period:i])
        donchian_low[i] = np.min(low[i-donchian_period:i])
    
    # Align all indicators to LTF (6h)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Weekly trend filter
        bullish_trend = close[i] > ema50_1w_aligned[i]
        bearish_trend = close[i] < ema50_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
        # Exit logic: opposite Donchian breakout (trend reversal signal)
        long_exit = short_breakout  # Exit long when price breaks below Donchian low
        short_exit = long_breakout  # Exit short when price breaks above Donchian high
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_donchian_breakout_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0