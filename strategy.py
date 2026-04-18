#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA200 filter and volume confirmation
# Donchian breakout captures momentum in trending markets
# 1d EMA200 filter ensures alignment with long-term trend (bullish above, bearish below)
# Volume spike (>2x 20-period average) confirms conviction
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakouts below lower band in downtrend)
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
name = "4h_Donchian20_1dEMA200_Volume"
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
    
    # Get 1d data for EMA200 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema200_1d_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band AND uptrend AND volume spike
            if close[i] > high_20[i] and uptrend and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian band AND downtrend AND volume spike
            elif close[i] < low_20[i] and downtrend and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band OR trend reverses
            if close[i] < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band OR trend reverses
            if close[i] > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals