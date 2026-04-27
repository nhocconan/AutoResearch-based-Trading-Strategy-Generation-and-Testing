#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Williams %R levels: oversold < -80, overbought > -20
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 60-period EMA on 1d for trend filter
    ema_60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    
    # 6h Williams %R (14-period) for entry timing
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    williams_r_6h[highest_high_6h == lowest_low_6h] = -50
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_60_1d_aligned[i]) or 
            np.isnan(williams_r_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # 1d trend filter: price above/below 60 EMA
        price_above_ema = close[i] > ema_60_1d_aligned[i]
        price_below_ema = close[i] < ema_60_1d_aligned[i]
        
        # 1d Williams %R conditions
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        
        # 6h Williams %R for entry timing: look for reversal from extreme
        wr_6h_oversold = williams_r_6h[i] < -80
        wr_6h_overbought = williams_r_6h[i] > -20
        
        # Long conditions: 1d oversold + 6h showing oversold reversal + volume
        long_setup = wr_oversold and wr_6h_oversold and price_above_ema and volume_filter[i]
        # Short conditions: 1d overbought + 6h showing overbought reversal + volume
        short_setup = wr_overbought and wr_6h_overbought and price_below_ema and volume_filter[i]
        
        if long_setup:
            signals[i] = 0.25
            position = 1
        elif short_setup:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite 1d Williams %R extreme
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

name = "6h_WilliamsR_WR_1dTrend_6hTiming_VolumeFilter"
timeframe = "6h"
leverage = 1.0