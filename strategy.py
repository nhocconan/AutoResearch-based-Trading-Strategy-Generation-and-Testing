#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide clear breakout levels from weekly structure.
# 1w EMA50 ensures alignment with higher timeframe trend to avoid counter-trend entries.
# Volume spike (>1.8x 50 EMA) confirms institutional participation.
# Discrete sizing 0.25 limits risk and reduces fee churn.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 30-80 trades over 4 years.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels from previous 20 days (using current data up to previous bar)
    # We need to calculate rolling max/min manually to avoid look-ahead
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute volume EMA for confirmation
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(ema50_aligned[i]) or np.isnan(vol_ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian high/low for previous 20 periods (excluding current bar)
        if i >= 20:
            # Look back 20 periods from previous bar (i-20 to i-1)
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            # Not enough data yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 50-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_50[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if close[i] > donchian_high and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif close[i] < donchian_low and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes
            donchian_mid = (donchian_high + donchian_low) / 2.0
            
            if (close[i] < donchian_mid or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes
            donchian_mid = (donchian_high + donchian_low) / 2.0
            
            if (close[i] > donchian_mid or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals