#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Williams %R (14-period) for overbought/oversold
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 60-period EMA for trend filter on 6h
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_60[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price above/below 60 EMA
        price_above_ema = close[i] > ema_60[i]
        price_below_ema = close[i] < ema_60[i]
        
        # Long conditions: oversold + price above EMA + volume
        long_signal = oversold and price_above_ema and volume_filter[i]
        # Short conditions: overbought + price below EMA + volume
        short_signal = overbought and price_below_ema and volume_filter[i]
        
        if long_signal:
            signals[i] = 0.25
            position = 1
        elif short_signal:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Williams %R extreme
        elif position == 1 and williams_r_aligned[i] > -20:
            signals[i] = 0.0
            position = 0
        elif position == -1 and williams_r_aligned[i] < -80:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR14_OversoldOverbought_60EMA_VolumeFilter"
timeframe = "6h"
leverage = 1.0