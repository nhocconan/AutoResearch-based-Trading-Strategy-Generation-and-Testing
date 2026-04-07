#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1w Trend Filter and Volume Confirmation
# Hypothesis: Donchian channel breakouts on 12h capture significant momentum moves.
# Combined with 1-week EMA200 trend filter to ensure trades align with higher timeframe trend.
# Volume confirmation ensures moves have institutional participation.
# Works in both bull and bear markets by only taking trades in direction of 1w trend.
# Targets 15-30 trades/year with disciplined entries to avoid overtrading.

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Donchian channel (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period high and low for Donchian channels
    high_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_12h, high_max)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_min)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for Donchian and volume SMA
        # Skip if required data not available
        if (np.isnan(ema200_12h[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or close[i] < ema200_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or close[i] > ema200_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high + volume confirmation + uptrend
            if (close[i] > donchian_high[i] and 
                vol_confirm and 
                close[i] > ema200_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low + volume confirmation + downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and 
                  close[i] < ema200_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals