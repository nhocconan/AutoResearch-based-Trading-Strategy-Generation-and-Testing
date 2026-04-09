#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
# - Uses 6h Donchian channel (20-period) for breakout signals
# - Confirms with 12h volume > 1.8x its 20-period average (strong participation)
# - Uses 1d EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Donchian breakouts capture strong momentum moves, volume filter reduces false breakouts
# - Trend filter ensures we trade in direction of higher timeframe trend, reducing whipsaw in ranging markets

name = "6h_12h_1d_donchian_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Volume > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (1.8 * avg_volume_20)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align HTF indicators to 6h
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and trend filter
            if (high[i] >= highest_high[i] and    # Break above Donchian high
                volume_spike_12h_aligned[i] and   # Volume confirmation
                close[i] > ema_50_1d_aligned[i]): # Trend filter (bullish)
                position = 1
                signals[i] = 0.25
            elif (low[i] <= lowest_low[i] and     # Break below Donchian low
                  volume_spike_12h_aligned[i] and # Volume confirmation
                  close[i] < ema_50_1d_aligned[i]): # Trend filter (bearish)
                position = -1
                signals[i] = -0.25
    
    return signals