#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume spike confirmation.
    # Weekly trend ensures we trade with the dominant market direction (works in bull/bear).
    # Volume spike filters low-confidence breakouts.
    # Donchian breakouts capture momentum; weekly trend + volume filter improves win rate.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 6h data for Donchian channels (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian(20) channels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume SMA(20) for volume spike filter
    volume_6h = df_6h['volume'].values
    volume_sma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_6h, volume_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below 50 EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume spike filter: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * volume_sma_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > upper_20_aligned[i]  # Break above upper band
        breakout_short = close[i] < lower_20_aligned[i]  # Break below lower band
        
        # Entry conditions: breakout with trend and volume confirmation
        long_entry = breakout_long and uptrend and volume_spike
        short_entry = breakout_short and downtrend and volume_spike
        
        # Exit conditions: price returns to opposite Donchian band
        long_exit = close[i] < lower_20_aligned[i]
        short_exit = close[i] > upper_20_aligned[i]
        
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

name = "6h_1w_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0