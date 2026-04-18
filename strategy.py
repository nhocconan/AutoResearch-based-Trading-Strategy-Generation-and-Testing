#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Donchian breakout provides trend-following entry, EMA50 on weekly ensures alignment with higher timeframe trend,
# volume spike (>2x 20-period average) confirms conviction. Works in both bull and bear markets by
# following the trend direction on higher timeframe. Target: 10-25 trades/year (40-100 total over 4 years).
name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 20-period Donchian channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 1w for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend AND volume spike
            if close[i] > high_max_20[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND volume spike
            elif close[i] < low_min_20[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower band OR trend reverses
            if close[i] < low_min_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper band OR trend reverses
            if close[i] > high_max_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals