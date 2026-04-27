#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when close > upper Donchian band (20-day high) and 1w EMA(34) rising and volume > 1.5x average.
# Short when close < lower Donchian band (20-day low) and 1w EMA(34) falling and volume > 1.5x average.
# Uses 1d timeframe for signal generation and 1w for trend filter to reduce false signals.
# Designed for ~10-20 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40  # 20 for Donchian + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: breakout above upper band, 1w uptrend, volume filter
        if (close[i] > high_max[i] and 
            ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and  # rising EMA
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: breakout below lower band, 1w downtrend, volume filter
        elif (close[i] < low_min[i] and 
              ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and  # falling EMA
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0