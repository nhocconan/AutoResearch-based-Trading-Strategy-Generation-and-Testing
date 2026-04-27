#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-week trend filter and volume confirmation
# Donchian breakout captures trend continuation, works in both bull and bear markets
# Weekly trend filter avoids counter-trend trades, volume confirms institutional interest
# Target: 15-25 trades per year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily data for Donchian channels (using 1d as proxy for 12h calculation)
    # We'll calculate Donchian on 12h data directly using rolling windows
    # But for simplicity and to follow patterns, we use higher timeframe for trend
    
    # Calculate 20-period Donchian channels on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Price breaks above upper Donchian + weekly uptrend + volume
        if (close[i] > high_20[i-1] and   # Break above previous period's high
            close[i] > ema50_1w_aligned[i] and  # Above weekly EMA50 (uptrend)
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Price breaks below lower Donchian + weekly downtrend + volume
        elif (close[i] < low_20[i-1] and    # Break below previous period's low
              close[i] < ema50_1w_aligned[i] and  # Below weekly EMA50 (downtrend)
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_DonchianBreakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0