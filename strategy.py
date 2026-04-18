#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour donchian breakout with volume confirmation and 12h trend filter
# Uses 20-period donchian channels to capture breakouts
# Volume > 2x 20-period average confirms breakout strength
# 12h EMA34 provides trend direction filter to avoid counter-trend trades
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 20-period donchian channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 12h EMA34
        uptrend = close[i] > ema34_12h_aligned[i]
        downtrend = close[i] < ema34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper donchian band AND uptrend AND volume spike
            if close[i] > high_max_20[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower donchian band AND downtrend AND volume spike
            elif close[i] < low_min_20[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower donchian band OR trend reverses
            if close[i] < low_min_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper donchian band OR trend reverses
            if close[i] > high_max_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals