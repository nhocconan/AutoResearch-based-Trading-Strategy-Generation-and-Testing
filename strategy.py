#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Donchian breakouts capture strong momentum moves in both bull and bear markets.
# Weekly EMA50 ensures we only trade in the direction of the higher-timeframe trend.
# Volume confirmation filters out false breakouts.
# Designed for ~15-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on daily timeframe
    # We'll calculate this on 1d data and align to lower timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian bands: 20-period high and low
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe
    dch_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    dch_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dch_high_20_aligned[i]) or np.isnan(dch_low_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: price breaks above Donchian upper band in uptrend with volume
        if (close[i] > dch_high_20_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            if position != 1:  # Only signal on new entry
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25  # Maintain position
        
        # Short signal: price breaks below Donchian lower band in downtrend with volume
        elif (close[i] < dch_low_20_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            if position != -1:  # Only signal on new entry
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25  # Maintain position
        
        # Exit signals: reverse signals or loss of trend/volume
        elif position == 1 and (close[i] < ema50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > ema50_1w_aligned[i] or not volume_filter[i]):
            signals[i] = 0.0
            position = 0
        
        # Hold existing position if no exit signal
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0