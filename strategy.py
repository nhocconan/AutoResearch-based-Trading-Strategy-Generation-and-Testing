#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day trend filter and volume confirmation
# Donchian channel breakout captures breakouts in both bull and bear markets
# Trend filter: 1-day EMA50 determines market bias - only trade breakouts in trend direction
# Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts
# Position size: 0.25 for trend-aligned breakouts, 0.0 otherwise
# Stop loss: exit when price retests the middle of Donchian channel
# Target: 20-50 total trades over 4 years (5-12.5/year) - low frequency to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channel on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend from daily EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Look for breakouts in direction of trend
            bullish_breakout = (price > highest_high[i]) and has_volume
            bearish_breakout = (price < lowest_low[i]) and has_volume
            
            # Only take breakouts that align with trend
            if bullish_breakout and is_uptrend:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and is_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price returns to Donchian middle (mean reversion within trend)
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to Donchian middle
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0